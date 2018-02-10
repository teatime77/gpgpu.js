Classes = [];
SourceFiles = [];

/*
    空白の終わりを探す。
*/
function SkipSpace(str, start) {
    var pos;
    for (pos = start; pos < str.length && str[pos] == ' '; pos++);

    return pos;
}

/*
    名前の終わりを探す。
*/
function SkipName(str, start) {
    var pos;
    for (pos = start; pos < str.length && " =;\n({".indexOf(str[pos]) == -1; pos++);

    return [ pos, str.substring(start, pos) ];
}

/*
    ソースファイルのクラス
*/
class SourceFile{
    constructor(file_name) {
        this.comment = null;
        this.fileName = file_name;

        this.classes = [];
        this.functions = [];
    }

    makeRst(fs, dir_path) {
        var rst = "";

        rst += this.fileName + "\n";
        rst += "=".repeat(this.fileName.length) + "\n\n";

        if (this.comment != null) {

            rst += "説明\n^^^^^^\n\n";
            rst += this.comment + "\n\n";
        }

        rst += `

.. toctree::
    :maxdepth: 1
    :caption: 関数:

`;

        for(let fnc of this.functions) {
            fnc.makeRst(fs, dir_path);

            rst += "    " + fnc.fncName + "\n";
        }


        rst += `

.. toctree::
    :maxdepth: 1
    :caption: クラス:

`;

        for(let cls of this.classes) {
            fs.mkdirsSync(dir_path + cls.className);
            cls.makeRst(fs, dir_path + cls.className + "/");

            rst += "    " + cls.className + "/index\n";
        }

        fs.writeFile(dir_path + 'index.rst', rst);
    }
}

/*
    クラス定義のクラス
*/
class Class {
    constructor(source_file, comment, indent, class_name, super_class_name) {
        this.sourceFile = source_file;
        this.comment = comment;
        this.className = class_name;
        this.superClassName = super_class_name;
        this.indent = indent;
        this.methods = [];

        console.log("class " + class_name + " " + super_class_name);
    }

    makeRst(fs, dir_path) {
        var rst = "";

        rst += this.className + "\n";
        rst += "=".repeat(this.className.length) + "\n\n";

        if (this.comment != null) {

            rst += "説明\n^^^^^^\n\n";
            rst += this.comment + "\n\n";
        }


        rst += `

.. toctree::
    :maxdepth: 1
    :caption: メソッド:

`;

        for(let fnc of this.methods) {
            fnc.makeRst(fs, dir_path);

            rst += "    " + fnc.fncName + "\n";
        }


        fs.writeFile(dir_path + 'index.rst', rst);
    }
}

/*
    関数定義のクラス
*/
class Function {
    /*
        コンストラクタ

        関数定義の開始行の情報を保存する。
    */
    constructor(comment, start_line_idx, line, indent, is_generator, fnc_name, parent_class) {
        this.comment = comment;
        this.startLineIdx = start_line_idx;
        this.indent = indent;
        this.isGenerator = is_generator;
        this.fncName = fnc_name;
        this.parentClass = parent_class;

        this.head = line.trim();
        if (this.head.endsWith("{")) {
            // 関数定義の開始行の末尾が"{"の場合

            // 末尾の"{"を取り除く。
            this.head = this.head.substring(0, this.head.length - 1);
        }

        console.log("メソッド " + this.fncName);
    }

    /*
        関数のヘッダーと説明とソースのドキュメントを作る。
    */
    makeRst(fs, dir_path) {

        var rst = "";
        rst += this.fncName + "\n";
        rst += "=".repeat(2 * this.fncName.length) + "\n\n";
//        rst += "構文\n^^^^^^\n\n";
        var class_name = (this.parentClass ? this.parentClass.className + "." : "");
        rst += ".. js:function:: " + class_name + this.head + "\n\n";

        if (this.comment != null) {

//            rst += "説明\n^^^^^^\n\n";
            rst += this.comment + "\n\n";
        }

        rst += "ソース\n^^^^^^\n\n";
        rst += ".. code-block:: js\n\n" + this.body + "\n\n";

        fs.writeFile(dir_path + this.fncName + '.rst', rst);
    }
}

/*
    ソースファイルの構文解析
*/
function parseSource(fs, file_name) {
    var source_file = new SourceFile(file_name);
    SourceFiles.push(source_file);

    fs.mkdirsSync('../docs/api/' + file_name);

    console.log(file_name);

    var buf = fs.readFileSync(file_name + ".js");
    var str = buf.toString().replace(/\r\n/g, "\n").replace(/\t/g, "    ").replace(/[ ]+$/g, "");

    var class_name = null;
    var class_indent = -1;
    var current_class = null;
    var current_fnc = null;
    var comment_start = null;
    var comment_indent;
    var comment = null;
    var prev_comment = null;

    var lines = str.split('\n');
    for (var line_idx = 0; line_idx < lines.length; line_idx++) {
        if (prev_comment != null) {

            comment = prev_comment;
            prev_comment = null;
        }
        else {
            comment = null;
        }


        var line = lines[line_idx];
        if (line.trim().length == 0) {
            // 空行の場合

            if (source_file.comment == null && comment != null) {

                source_file.comment = comment;
            }

            continue;
        }

        var indent = SkipSpace(line, 0);

        if (comment_start != null) {

            if (line.startsWith("*/", indent)) {
                prev_comment = lines.slice(comment_start + 1, line_idx).map(x => x.substring(comment_indent)).join("\n");
                comment_start = null;
            }

            continue;
        }

        if (line.startsWith("class", indent)) {
            // クラスの場合

            var pos = SkipSpace(line, indent + 5);
            [pos, class_name] = SkipName(line, pos);

            pos = SkipSpace(line, pos);

            var super_class_name = null;
            if (line.startsWith("extends", pos)) {

                pos = SkipSpace(line, pos + 7);

                [pos, super_class_name] = SkipName(line, pos);
            }

            current_class = new Class(source_file, comment, indent, class_name, super_class_name);
            source_file.classes.push(current_class);
            Classes.push(current_class);
        }
        else if (line.startsWith("/**", indent) || line.startsWith("/*", indent)) {
            // ブロックコメントの場合

            var s = line.trim();
            if (s.endsWith("*/")) {
                
                comment_indent = 0;
                prev_comment = s.substring(2, s.length - 2);
            }
            else {

                comment_indent = indent;
                comment_start = line_idx;
            }
        }
        else if (line.startsWith("//", indent)) {
            // 行コメントの場合

            prev_comment = line.substring(indent + 2).trim();
        }
        else if (line.startsWith("function", indent)) {
            // 関数の場合

            var pos = SkipSpace(line, indent + 8);
            var fnc_name;

            [pos, fnc_name] = SkipName(line, pos);

            current_fnc = new Function(comment, line_idx, line, indent, false, fnc_name, null);
            source_file.functions.push(current_fnc);
        }
        else if (line.startsWith("constructor", indent)) {
            // コンストラクタの場合

            current_fnc = new Function(comment, line_idx, line, indent, false, "constructor", current_class);
            current_class.methods.push(current_fnc);
        }
        else if (line.startsWith("var", indent)) {
            // 変数の場合

        }
        else if (line.startsWith("}", indent)) {
            // ブロックの終わりの場合

            if (current_class != null && current_class.indent == indent) {

                current_class = null;
            }
            else if (current_fnc != null && current_fnc.indent == indent) {

                current_fnc.body = "    " + lines.slice(current_fnc.startLineIdx, line_idx + 1).join("\n    ");

                current_fnc = null;
            }

        }
        else {

            if (current_class != null && current_class.indent + 4 == indent && current_fnc == null) {
                // クラス定義の中で、クラス定義の直下のインデントで、関数定義の中でない場合

                var pos, fnc_name, is_generator = false;

                if (line[indent] == "*") {
                    // ジェネレーターの場合

                    is_generator = true;

                    pos = SkipSpace(line, indent + 1);
                }
                else {
                    // ジェネレーターでない場合

                    pos = indent;
                }

                [pos, fnc_name] = SkipName(line, pos);

                current_fnc = new Function(comment, line_idx, line, indent, is_generator, fnc_name, current_class);
                current_class.methods.push(current_fnc);
            }

        }
    }
}


// JavaScript source code
var fs = require('fs-extra');//"fs");

for(let file_name of["gpgpu", "util", "shader", "network", "plot", "MNIST", "make_rst"]) {
    parseSource(fs, file_name);

    var rst = `
リファレンス
===============

.. toctree::
    :maxdepth: 1
    :caption: モジュール:

`;


    for(let source_file of SourceFiles) {
        source_file.makeRst(fs, '../docs/api/' + source_file.fileName + "/");

        rst += "    " + source_file.fileName + "/index\n";
    }

    fs.writeFile('../docs/api/index.rst', rst);
}

