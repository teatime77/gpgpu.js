
class TTerm {
    TermSrc() {
        return "TERM";
    }
}

class TLiteral extends TTerm {
    constructor(tkn) {
        super();
        this.Token = tkn;
    }

    TermSrc() {
        return this.Token.TextTkn;
    }
}

class TReference extends TTerm {
    constructor(tkn) {
        super();
        this.Token = tkn;
    }

    TermSrc() {
        if (this.Token.TextTkn == "self") {

            return "this";
        }
        else {

            return this.Token.TextTkn;
        }
    }
}

class TDot extends TTerm {
    constructor(trm, tkn) {
        super();
        this.TermDot = trm;
        this.TokenDot = tkn;
    }

    TermSrc() {
        return this.TermDot.TermSrc() + "." + this.TokenDot.TextTkn;
    }
}

class TApply extends TTerm {
    constructor(trm, arg) {
        super();
        this.Fnc = trm;
        this.Arg = arg;
        arg.Parent = this;
    }

    TermSrc() {
        return this.Fnc.TermSrc() + this.Arg.TermSrc();
    }
}

class TList extends TTerm {
    constructor() {
        super();
    }

    TermSrc() {
        return "LIST";
    }
}

class TOperation extends TTerm {
    constructor(op_tkn, list) {
        super();
        this.Operator = op_tkn;
        this.List = list;
    }

    push(x) {
        this.List.push(x);
    }

    TermSrc() {
        if (this.List.length == 1) {

            return this.Operator.TextTkn + this.List[0].TermSrc();
        }
        else {
            var s = "";

            for (var i = 0; i < this.List.length; i++) {

                if (i != 0) {

                    s += " " + this.Operator.TextTkn + " ";
                }
                Assert(this.List[i] != null, "");
                s += this.List[i].TermSrc();
            }

            return s;
        }
    }
}

class TSlice extends TOperation {
    constructor(op_tkn, list) {
        super(op_tkn, list);
    }

    TermSrc() {
        return "[" + this.List.map(x => x == null ? "null" : x.TermSrc()).join(", ") + "]";
    }
}

class TTargetList extends TTerm {
    constructor(list) {
        super();
        this.List = list;
        var self = this;
        list.forEach(x => { Assert(x instanceof TTerm, "set parent"); x.Parent = self; });
    }

    TermSrc() {
        return "Target-List";
    }
}

class TExpressionListGenerator extends TTerm {
    constructor(list) {
        super();
        this.List = list;
    }

    TermSrc() {
        Assert(this.Parent, "Expression-List-Generator:Term-Src");
        if (this.List.length == 1) {
            return this.List[0].TermSrc();
        }
        else {
            var s = "[";
            for (var i = 0; i < this.List.length; i++) {
                if (i != 0) {
                    s += ", ";
                }
                s += this.List[i].TermSrc();
            }
            s += "]";

            return s;
        }
    }
}

class TCompFor extends TTerm {
    constructor(target_list, source) {
        super();
        Assert(target_list && target_list.List);
        this.TargetList = target_list;
        this.Source = source;
    }

    NestedFor(tab) {
        var list = this.TargetList.List;

        var s1;
        var s2 = tab + "}\r\n";

        if (list.length == 1) {
            s1 = tab + "for (let " + list[0].TermSrc() + " of " + this.Source.TermSrc() + "){\r\n";
        }
        else {

            var tmp_var = "$";
            s1 = tab + "for (let $ of " + this.Source.TermSrc() + "){\r\n";

            for (var i = 0; i < list.length; i++) {
                s += tab + "    var " + list[i].TermSrc() + " = $[" + i + "];\r\n";
            }
        }

        var s3;
        if(this.InnerFor){
            s3 = this.InnerFor.NestedFor(tab + "    ");
        }
        else{

            s3 = tab + " ".repeat(4) + "yield " + this.ElementValue.TermSrc() + ";\r\n";
        }

        return s1 + s3 + s2;
    }

    TermSrc() {
        var s;

        Assert(this.ElementValue, "Comp-For:Term-Src");

        if (this.InnerFor) {
            s = "Array.from(function* () {\r\n";

            s += this.NestedFor("    ");
            return s + "}())";
        }
        else {

            var list = this.TargetList.List;

            s = this.Source.TermSrc() + ".map(";
            if (list.length == 1 && list[0] instanceof TReference) {
                return s + list[0].TermSrc() + " => " + this.ElementValue.TermSrc() + ")";
            }
            else {
                var params;

                if (list.length == 1) {

                    Assert(list[0] instanceof TParenthesizedFormGenerator, "Comp-For:Term-Src:params");
                    params = list[0].List;
                }
                else {

                    params = list;
                }

                var tmp_var = "$";
                s += "$ => {";
                for (var i = 0; i < params.length; i++) {
                    s += "var " + params[i].TermSrc() + " = $[" + i + "];";
                }
                s += "return " + this.ElementValue.TermSrc() + ";})";
                return s;
            }
        }
    }
}

class TBracket extends TTerm {
    constructor(left_term, list, comp_for) {
        super();
        this.LeftTerm = left_term;
        this.List = list;
        this.CompFor = comp_for;
    }

    TermSrc() {
        var s = "";

        if (this.LeftTerm) {

            s = this.LeftTerm.TermSrc();
        }

        if (this.CompFor) {

            return s + this.CompFor.TermSrc();
        }
        else {

            if (this.List.length != 0 && this.List[0] instanceof TSlice) {

                return "Slice(" + s + ", " + this.List.map(x => x.TermSrc()).join(", ") + ")";
            }
            else {
                if (this.LeftTerm) {

                    return s + ".GetAt(" + this.List.map(x => x.TermSrc()).join(", ") + ")";
                }
                else {

                    return "[" + this.List.map(x => x.TermSrc()).join(", ") + "]";
                }

            }
        }
    }
}

class TParenthesizedFormGenerator extends TTerm {
    constructor(list, is_tuple, comp_for) {
        super();
        this.List = list;
        this.IsTuple = is_tuple;
        this.CompFor = comp_for;
    }

    TermSrc() {
        if (this.CompFor) {

            return "(" + this.CompFor.TermSrc() + ")";
        }
        else {

            if (this.Parent instanceof TApply || this.List.length == 1 && ! this.IsTuple) {

                return "(" + this.List.map(x => x.TermSrc()).join(", ") + ")";
            }
            else {

                return "[" + this.List.map(x => x.TermSrc()).join(", ") + "]";
            }
        }
    }
}

class TSetDictionary extends TTerm {
    constructor(list) {
        super();
        this.List = list;
    }

    TermSrc() {
        var s = "{";
        for (var i = 0; i < this.List.length; i++) {
            if (i != 0) {
                s += ", ";
            }
            s += this.List[i].toString();
        }
        s += "}";

        return s;
    }
}

class TKeyValue {
    constructor(key, val) {
        this.Key = key;
        this.Value = val;
    }

    toString() {
        return "\"" + this.Key + "\":" + this.Value.TermSrc();
    }
}

/*

class  extends TTerm {
}

class  extends TTerm {
}
*/

class TStatement {
    Src(tab) {
        return "<??>";
    }
}

class TBlockStatement extends TStatement {
    constructor(statement_end) {
        super();
        if (statement_end != null) {

            this.Statements = statement_end;
        }
        else {

            this.Statements = new Array();
        }
    }

    Src(tab) {
        return "<????>";
    }

    BlockSrc(tab) {
        var tab2 = tab + "    ";
        var src = "";
        for(let stmt of this.Statements) {
            src += stmt.Src(tab2);
        }

        return src;
    }
}

class TElseIf extends TBlockStatement {
    constructor() {
        super(null);
    }

    Src(tab) {
        return tab + "else if(){\r\n" +
                this.BlockSrc(tab) +
            tab + "}\r\n";
    }
}

class TElse extends TBlockStatement {
    constructor() {
        super(null);
    }

    Src(tab) {
        return tab + "else{\r\n" +
            this.BlockSrc(tab) +
        tab + "}\r\n";
    }
}

class TFor extends TBlockStatement {
    constructor(comp_for, statement_end) {
        super(statement_end);
        this.CompFor = comp_for;
    }

    Src(tab) {
        var target_list = this.CompFor.TargetList.List;
        var s = tab;

        if (target_list.length == 1) {

            s += "for (let " + target_list[0].TermSrc() + " of " + this.CompFor.Source.TermSrc() + ") {\r\n";
        }
        else {

            var loop_var_name = "$" + (tab.length/4);
            s += "for (let " + loop_var_name + " of " + this.CompFor.Source.TermSrc() + ") {\r\n";
            for (var i = 0; i < target_list.length; i++) {
                s += tab + "    var " + target_list[i].TermSrc() + " = " + loop_var_name + "[" + i + "];\r\n";
            }
        }

        return s +
            this.BlockSrc(tab) +
            tab + "}\r\n";
    }
}

class TIf extends TBlockStatement {
    constructor(cnd, statement_end) {
        super(statement_end);
        this.Condition = cnd;
    }

    Src(tab) {
        return tab + "if(" + this.Condition.TermSrc() + "){\r\n" +
            this.BlockSrc(tab) +
        tab + "}\r\n";
    }
}

class TTry extends TBlockStatement {
    constructor() {
        super(null);
    }

    Src(tab) {
        return tab + "try{\r\n" +
            this.BlockSrc(tab) +
        tab + "}\r\n";
    }
}

class TExcept extends TBlockStatement {
    constructor() {
        super(null);
    }

    Src(tab) {
        return tab + "catch(e){\r\n" +
            this.BlockSrc(tab) +
        tab + "}\r\n";
    }
}

class TImport extends TStatement {

    Src(tab) {
        return tab + "// import\r\n";
    }
}

class TReturn extends TStatement {
    constructor(ret_val) {
        super();
        this.RetVal = ret_val;

        if (ret_val != null) {

            ret_val.Parent = this;
        }
    }

    Src(tab) {
        if(this.RetVal == null){

            return tab + "return;\r\n";
        }
        else{

            return tab + "return " +this.RetVal.TermSrc() + ";\r\n";
        }
    }
}

class TPrint extends TStatement {
    constructor(print_val) {
        super();
        this.PrintVal = print_val;
        if (print_val != null) {

            print_val.Parent = this;
        }
    }

    Src(tab) {
        if (this.PrintVal == null) {

            return tab + "console.log();\r\n";
        }
        else {

            return tab + "console.log(" + this.PrintVal.TermSrc() + ");\r\n";
        }
    }
}

class TPass extends TStatement {

    Src(tab) {
        return tab + ";\r\n";
    }
}

class TAssignment extends TStatement {

    constructor(target_list, val) {
        super();
        this.TargetList = target_list;
        this.Value = val;

        target_list.Parent = this;
        val.Parent = this;
    }

    Src(tab) {
        var list = this.TargetList.List;
        var s = tab;

        if(list.length == 1){

            if (list[0] instanceof TBracket) {
                var bracket = list[0];

                Assert(bracket.List && bracket.List.length == 1, "Set At");
                s += "SetAt(" + bracket.LeftTerm.TermSrc() + ", " + bracket.List[0].TermSrc() + ", " + this.Value.TermSrc() + ");\r\n";
            }
            else {

                s += list[0].TermSrc() + "=" + this.Value.TermSrc() + ";\r\n";
            }
        }
        else{

            s += "$ =" + this.Value.TermSrc() + ";\r\n";
            for(var i = 0; i < list.length; i++){

                s += tab + list[i].TermSrc() + " = $[" + i + "];\r\n";
            }
        }
        return s;
    }
}

class TCall extends TStatement {
    constructor(trm1) {
        super();
        this.TermCall = trm1;
        trm1.Parent = this;
    }

    Src(tab) {
        return tab + this.TermCall.TermSrc() + ";\r\n";
    }
}

class TDecorator extends TStatement {

    Src(tab) {
        return "";
    }
}

class TFunction {
    constructor(fnc_name, params) {
        this.FunctionName = fnc_name;
        this.Params = params;
        this.Statements = new Array();
        this.InnerFunctions = new Array();
    }

    FunctionSrc(tab) {

        var s = tab;

        if (!this.Parent) {

            s += "function ";
        }

        var fnc_name = (this.FunctionName == "__init__" ? "constructor" : this.FunctionName);

        var params = (this.Params.length != 0 && this.Params[0].VarName.TextTkn == "self" ? this.Params.slice(1) : this.Params);
        s += fnc_name + "(" + params.map(x => x.VarSrc()).join(", ") + "){\r\n";
        for(let var1 of this.Params) {
            if (var1.DefaultValue) {
                s += tab + "    if(" + var1.VarSrc() + " == undefined){ " + var1.VarSrc() + " = " + var1.DefaultValue.TermSrc() + ";}\r\n";
            }
        }
        for(var stmt of this.Statements) {
            s += stmt.Src(tab + "    ");
        }
        s += tab + "}\r\n";
        return s;
    }
}

class TVariable {
    constructor(var_name) {
        this.VarName = var_name;
    }

    VarSrc() {
        return this.VarName.TextTkn;
    }
}

class TClass {
    constructor(class_name) {
        this.ClassName = class_name;
        this.Functions = new Array();
    }

    ClassSrc() {

        var s = "class " + this.ClassName + " {\r\n";
        for(var fnc of this.Functions) {
            s += fnc.FunctionSrc("    ");
        }
        s += "}\r\n";

        return s;
    }
}


class TSource {
    constructor() {
        this.CodeList = new Array();
    }

    SourceSrc() {

        var s = "";
        for(var x of this.CodeList) {
            if (x instanceof TClass) {

                s += x.ClassSrc();
            }
            else if (x instanceof TFunction) {

                s += x.FunctionSrc("");
            }
            else if (x instanceof TStatement) {

                s += x.Src("");
            }
            else {

                Assert(false, "Code-List");
            }
        }

        return s;
    }

    MakeHTML() {
        var txt = "";
        for (var i = 0; i < this.Lines.length; i++) {
            var line = this.Lines[i];

            if (line.Tokens.length != 0) {
                var pos = 0;
                for (let tkn of line.Tokens) {
                    if (pos < tkn.StartPos) {
                        txt += " ".repeat(tkn.StartPos - pos);
                    }
                    var span_class = "";
                    switch (tkn.TokenType) {
                        case ETokenType.String_:
                        case ETokenType.VerbatimString:
                        case ETokenType.VerbatimStringContinued:
                            span_class = "string";
                            break;

                        case ETokenType.LineComment:
                            span_class = "comment";
                            break;

                        case ETokenType.Keyword:
                            span_class = "keyword";
                            break;

                        case ETokenType.Identifier:
                            if (tkn.TextTkn == "self" || tkn.TextTkn == "True" || tkn.TextTkn == "False") {
                                span_class = "keyword";

                            }
                            break;
                    }
                    txt += "<span class='" + span_class + "'>" + tkn.TextTkn + "</span>";

                    pos = tkn.EndPos;
                }
            }
            txt += "\r\n";
        }

        return txt;
    }
}
