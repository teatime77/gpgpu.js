class TLexer {
    constructor() {
        //this.KeywordMap = new Object();
        this.KeywordMap = {
            "False": EKind.False_,
            "None": EKind.None_,
            "True": EKind.True_,
            "and": EKind.and_,
            "as": EKind.as_,
            "assert": EKind.assert_,
            "break": EKind.break_,
            "class": EKind.class_,
            "continue": EKind.continue_,
            "def": EKind.def_,
            "del": EKind.del_,
            "elif": EKind.elif_,
            "else": EKind.else_,
            "except": EKind.except_,
            "finally": EKind.finally_,
            "for": EKind.for_,
            "from": EKind.from_,
            "global": EKind.global_,
            "if": EKind.if_,
            "import": EKind.import_,
            "in": EKind.in_,
            "is": EKind.is_,
            "lambda": EKind.lambda_,
            "nonlocal": EKind.nonlocal_,
            "not": EKind.not_,
            "or": EKind.or_,
            "pass": EKind.pass_,
            "print": EKind.print_,
            "raise": EKind.raise_,
            "return": EKind.return_,
            "try": EKind.try_,
            "while": EKind.while_,
            "with": EKind.with_,
            "yield": EKind.yield_,
        };
        //for (var x in this.KeywordMap) {
        //    console.log("%s %s", x, EnumName(EKind, this.KeywordMap[x]));
        //}

        this.SymbolTable = {
            "**": EKind.Power,
            "+": EKind.Add,
            "-": EKind.Sub,
            "*": EKind.Mul,
            "**": EKind.Power,
            "/": EKind.Div,
            "//": EKind.FloorDiv,
            "%": EKind.Mod_,
            "@": EKind.At,
            "<<": EKind.LeftShift,
            ">>": EKind.RightShift,
            "&": EKind.Anp,
            "|": EKind.BitOR,
            "^": EKind.Hat,
            "~": EKind.Tilde,
            "<": EKind.LT,
            ">": EKind.GT,
            "<=": EKind.LE,
            ">=": EKind.GE,
            "==": EKind.Eq,
            "!=": EKind.NE,
            "(": EKind.LP,
            ")": EKind.RP,
            "[": EKind.LB,
            "]": EKind.RB,
            "{": EKind.LC,
            "}": EKind.RC,
            ",": EKind.Comma,
            ":": EKind.Colon,
            ".": EKind.Dot,
            ";": EKind.SemiColon,
            "=": EKind.Assign,
            //"->": EKind.,
            "+=": EKind.AddEq,
            "-=": EKind.SubEq,
            "*=": EKind.MulEq,
            "/=": EKind.DivEq,
            "\\": EKind.Backslash,
            //"//=": EKind.,
            "%=": EKind.ModEq,
            "@=": EKind.AtEq,
            //"&=": EKind.,
            //"|=": EKind.,
            //"^=": EKind.,
            "<<=": EKind.LeftShiftEq,
            ">>=": EKind.RightShiftEq,
            "**=": EKind.PowerEq,
        };

        //for (var x in this.SymbolTable) {
        //    console.log("%s %s", x, EnumName(EKind, this.SymbolTable[x]));
        //}
    }

    IsWhiteSpace(ch){
        return " \t\r\n\f\v".indexOf(ch) != -1;
    }

    IsDigit(ch){
        return "0123456789".indexOf(ch) != -1;
    }

    /*
        16進数文字ならtrue
    */
    IsHexDigit(ch) {
        return "0123456789abcdefABCDEF".indexOf(ch) != -1;
    }

    IsLetter(ch){
        if (!this.LetterTable) {

            this.LetterTable = GetLetterTable();
        }
        
        return this.LetterTable[ch.charCodeAt(0)] != 0;
    }

    IsLetterOrDigit(ch){
        return this.IsDigit(ch) || this.IsLetter(ch);
    }

    /*
        エスケープ文字を読み込み、文字位置(pos)を進める。
    */
    ReadEscapeString(text, start_pos, end_pos) {
        var s = "";

        for(pos = start_pos; pos < end_pos;){
            var ch1 = text[pos + 1];

            if(ch1 != '\\'){
                // エスケープ文字でない場合

                s += ch1;
                pos++;
            }
            else{
                // エスケープ文字の場合

                // 1文字のエスケープ文字の変換リスト
                var in_str = "\'\"\\0abfnrtv";
                var out_str = "\'\"\\\0\a\b\f\n\r\t\v";

                // 変換リストにあるか調べる。
                var k = in_str.indexOf(text[pos + 1]);

                if (k != -1) {
                    // 変換リストにある場合

                    pos += 2;

                    // 変換した文字を返す。
                    s += out_str[k];
                }
                else{
                    // 変換リストにない場合

                    switch (text[pos + 1]) {
                    case 'u':
                        // \uXXXX

                        pos = Math.Min(pos + 6, end_pos);

                        // エスケープ文字の計算は未実装。
                        break;

                    case 'U':
                        // \UXXXXXXXX

                        pos = Math.Min(pos + 10, end_pos);

                        // エスケープ文字の計算は未実装。
                        break;

                    case 'x':
                        // \xX...

                        // 16進数字の終わりを探す。
                        for (pos++; pos < end_pos && this.IsHexDigit(text[pos]); pos++) ;

                        // エスケープ文字の計算は未実装。
                        break;

                    default:
                        // 上記以外のエスケープ文字の場合

                        Debug.WriteLine("Escape Sequence Error  [{0}]", text[pos + 1]);

                        pos += 2;
                        break;
                    }
                }
            }
        }

        return s;
    }

    /*
        字句解析をして各文字の字句型の配列を得る。
    */
    LexicalAnalysis(text, prev_token_type) {
        var token_list = new Array();

        // 文字列の長さ
        var text_len = text.length;

        // 現在の文字位置
        var pos = 0;

        // 各文字の字句型の配列
        //ETokenType[] token_type_list = new ETokenType[text_len];

        // 文字列の最後までループする。
        while (pos < text_len) {
            var token_kind = EKind.Undefined;
            var token_type = ETokenType.Error;

            // 字句の開始位置
            var start_pos = pos;

            // 現在位置の文字
            var ch1 = text[pos];
            var s2 = text.substr(pos, 2);
            var s3 = text.substr(pos, 3);

            var is_white = false;
            var escape_str = null;
            if (pos == 0 && prev_token_type == ETokenType.VerbatimStringContinued) {
                // 文字列の最初で直前が閉じてないブロックコメントか逐語的文字列の場合

                token_kind = EKind.StringLiteral;

                // 逐語的文字列の終わりを探す。
                var k = text.indexOf('\"\"\"');

                if (k != -1) {
                    // 逐語的文字列の終わりがある場合

                    token_type = ETokenType.VerbatimString;
                    pos = k + 3;
                }
                else {
                    // 逐語的文字列の終わりがない場合

                    token_type = ETokenType.VerbatimStringContinued;
                    pos = text_len;
                }
            }
            else if (this.IsWhiteSpace(ch1)) {
                // 空白の場合

                is_white = true;

                // 空白の終わりを探す。
                for (pos++; pos < text_len && this.IsWhiteSpace(text[pos]) ; pos++);
            }
            else if (s3 == "\"\"\"") {
                // 逐語的文字列の場合

                token_kind = EKind.StringLiteral;

                // 逐語的文字列の終わりの位置
                var k = text.indexOf('\"\"\"', pos + 2);

                if (k != -1) {
                    // 逐語的文字列の終わりがある場合

                    token_type = ETokenType.VerbatimString;
                    pos = k + 3;
                }
                else {
                    // 逐語的文字列の終わりがない場合

                    token_type = ETokenType.VerbatimStringContinued;
                    pos = text_len;
                }
            }
            else if (this.IsLetter(ch1) || ch1 == '_') {
                // 識別子の最初の文字の場合

                // 識別子の文字の最後を探す。識別子の文字はユニコードカテゴリーの文字か数字か'_'。
                for (pos++; pos < text_len && (this.IsLetterOrDigit(text[pos]) || text[pos] == '_') ; pos++);

                // 識別子の文字列
                var name = text.substring(start_pos, pos);

                token_kind = this.KeywordMap[name];
                if (token_kind != undefined) {
                    // 名前がキーワード辞書にある場合

                    token_type = ETokenType.Keyword;
                }            
                else {
                    // 名前がキーワード辞書にない場合

                    token_kind = EKind.Identifier;
                    token_type = ETokenType.Identifier;
                }
            }
            else if (this.IsDigit(ch1)) {
                // 数字の場合

                token_kind = EKind.NumberLiteral;

                if (s2 == "0x") {
                    // 16進数の場合

                    pos += 2;

                    // 16進数字の終わりを探す。
                    for (; pos < text_len && this.IsHexDigit(text[pos]) ; pos++);

                    token_type = ETokenType.Int;
                }
                else {
                    // 10進数の場合

                    // 10進数の終わりを探す。
                    for (; pos < text_len && this.IsDigit(text[pos]) ; pos++);

                    if (pos < text_len && text[pos] == '.') {
                        // 小数点の場合

                        pos++;

                        // 10進数の終わりを探す。
                        for (; pos < text_len && this.IsDigit(text[pos]) ; pos++);

                        if (pos < text_len && text[pos] == 'f') {
                            // floatの場合

                            pos++;
                            token_type = ETokenType.Float;
                        }
                        else {
                            // doubleの場合

                            token_type = ETokenType.Double;
                        }
                    }
                    else {

                        token_type = ETokenType.Int;
                    }
                }
            }
            else if (ch1 == '\"' || ch1 == '\'') {
                // 文字列の場合

                var end_char = ch1;
                token_kind = EKind.StringLiteral;
                token_type = ETokenType.Error;

                var has_escape = false;
                // 文字列の終わりを探す。
                for (pos++; pos < text_len;) {
                    var ch3 = text[pos];

                    if (ch3 == end_char) {
                        // 文字列の終わりの場合

                        // ループを抜ける。
                        pos++;
                        token_type = ETokenType.String_;
                        break;
                    }
                    else if (ch3 == '\\') {
                        // エスケープ文字の場合

                        // 文字位置(pos)を2つ進める。
                        pos += 2;
                    }
                    else {
                        // エスケープ文字でない場合

                        pos++;
                    }
                }
                if(has_escape){
                    // エスケープ文字がある場合

                    escape_str = this.ReadEscapeString(text, start_pos, pos);
                }
                else{
                    // エスケープ文字がない場合

                }
            }
            else if (ch1 == '#') {
                // 行コメントの場合

                // 空白か行の先頭までstart_posを戻す。
                while (0 < start_pos && this.IsWhiteSpace(text[start_pos - 1])) {
                    start_pos--;
                }

                token_kind = EKind.LineComment;
                token_type = ETokenType.LineComment;

                // 改行を探す。
                var k = text.indexOf('\n', pos);

                if (k != -1) {
                    // 改行がある場合

                    pos = k;
                }
                else {
                    // 改行がない場合

                    pos = text_len;
                }
            }
            else {
                // 不明の文字の場合

                token_type = ETokenType.Symbol;

                token_kind = this.SymbolTable[s3];
                if (token_kind) {

                    pos += 3;
                }
                else{

                    token_kind = this.SymbolTable[s2];
                    if (token_kind) {

                        pos += 2;
                    }
                    else{

                        token_kind = this.SymbolTable[ch1];
                        if (token_kind) {

                            pos += 1;
                        }
                        else{

                            token_type = ETokenType.Error;
                            token_kind = EKind.Undefined;
                            pos++;
                        }
                    }
                }
            }

            if (!is_white) {
                // 空白でない場合

                // 字句の文字列を得る。
                var s;

                if (escape_str != null) {
                    s = escape_str;
                }
                else {

                    s = text.substring(start_pos, pos);
                }

                // トークンを作り、トークンのリストに追加する。
                token_list.push(new TToken(token_type, token_kind, s, start_pos, pos));
            }
        }

        // 各文字の字句型の配列を返す。
        return token_list;
    }
}

class TLine {
    constructor(text, tokens) {
        this.Text = text;
        this.Tokens = tokens;
    }
}

class TToken {
/*
    var TokenType;
    var Kind;
    var TextTkn;
    var StartPos;
    var EndPos;
    var ErrorTkn;
    var TabTkn;
    var ObjTkn;

    TToken() {
    }

    TToken(TToken token) {
        TokenType   = token.TokenType;
        Kind        = token.Kind;
        TextTkn     = token.TextTkn;
        StartPos    = token.StartPos;
        EndPos      = token.EndPos;
        ErrorTkn    = token.ErrorTkn;
        TabTkn      = token.TabTkn;
        ObjTkn      = token.ObjTkn;
    }

    TToken(object obj) {
        ObjTkn = obj;
    }

    TToken(EKind kind) {
        Kind = kind;
    }

    TToken(EKind kind, object obj) {
        Kind = kind;
        ObjTkn = obj;
    }

    TToken(string txt, object obj) {
        TextTkn = txt;
        ObjTkn = obj;
    }
*/

    constructor(token_type, kind, txt, start_pos, end_pos) {
        this.TokenType = token_type;
        this.Kind = kind;
        this.TextTkn = txt;
        this.StartPos = start_pos;
        this.EndPos = end_pos;
    }
}


class TParseException {
    constructor(parser) {
        var s = "";
        var pos = 0;

        for(let tkn of parser.TokenList) {
            if (pos < tkn.StartPos) {
                s += " ".repeat(tkn.StartPos - pos);
            }
            if (tkn == parser.CurrentToken) {

                s += " ^ ";
            }
            s += tkn.TextTkn;
            pos = tkn.EndPos;
        }
        console.log("parse error! " + (parser.LineIdx + 1) + " " + s);
    }
}
