// JavaScript source code
class TParser {

    constructor() {
        Number.prototype.add = function (x) {
            return this + x;
        };
        Number.prototype.mul = function (x) {
            return this * x;
        };
        var x, y = 0.1;

        var startTime;

        x = 1;
        startTime = new Date();
        for (var i = 0; i < 1000000; i++) {
            x = x + y;
        }
        console.log("%f %dms", x, new Date() - startTime);

        x = 1;
        startTime = new Date();
        for (var i = 0; i < 1000000; i++) {
            x = x.add(y);
        }
        console.log("%f %dms", x, new Date() - startTime);

        x = 1;
        startTime = new Date();
        for (var i = 0; i < 1000000; i++) {
            x = x * y;
        }
        console.log("%f %dms", x, new Date() -startTime);

        x = 1;
        startTime = new Date();
        for(var i = 0; i < 1000000; i++) {
            x = x.mul(y);
        }
        console.log("%f %dms", x, new Date() -startTime);


        console.log("1    +2:%.1f  +2.3:%.1f  +10:%.1f", x.add(2), x.add(2.3), x.add(10));
        x = 2.3;
        console.log("2.3  +2:%.1f  +2.3:%.1f  +10:%.1f", x.add(2), x.add(2.3), x.add(10));
        this.EOTToken = new TToken(ETokenType.EOT, EKind.EOT, "", 0, 0);
    }

    MakeOperation1(op_tkn, trm1) {
        return new TOperation(op_tkn, [trm1]);
    }

    MakeOperation2(op_tkn, trm1, trm2) {
        return new TOperation(op_tkn, [trm1, trm2]);
    }

    // proper_slice ::=  [lower_bound] ":" [upper_bound] [ ":" [stride] ]
    ProperSlice() {
        var slice = null;
        var trm1 = null;
        var op_tkn;

        while (this.CurrentToken.Kind != EKind.Comma && this.CurrentToken.Kind != EKind.RB) {
            if (this.CurrentToken.Kind == EKind.Colon) {

                op_tkn = this.GetToken(EKind.Colon);
                if (slice == null) {

                    slice = new TSlice(op_tkn, new Array());
                }
                slice.push(null);
            }
            else {

                trm1 = this.Expression();
                if (this.CurrentToken.Kind == EKind.Colon) {

                    op_tkn = this.GetToken(EKind.Colon);
                    if (slice == null) {

                        slice = new TSlice(op_tkn, new Array());
                    }
                    slice.push(trm1);
                }
                else{

                    if (slice != null) {

                        slice.push(trm1);
                    }
                    break;
                }
            }
        }

        return slice != null ? slice : trm1;
    }

    // subscription ::=  primary "[" expression_list "]"
    // slicing ::=  primary "[" slice_list "]"
    SubscriptionSlicingCompFor(left_term) {
        this.GetToken(EKind.LB);

        if(this.CurrentToken.Kind == EKind.RB){

            this.GetToken(EKind.RB);

            return new TBracket(left_term, new Array(), null);
        }

        var list = new Array();

        while (true) {
            var trm1 = this.ProperSlice();

            if(this.CurrentToken.Kind == EKind.for_){

                Assert(left_term == null, "Comp-For");

                var comp_for = this.CompFor(trm1);

                this.GetToken(EKind.RB);

                return new TBracket(left_term, null, comp_for);
            }

            list.push(trm1);

            if(this.CurrentToken.Kind == EKind.RB){

                break;
            }
            this.GetToken(EKind.Comma);
        }

        this.GetToken(EKind.RB);
        return new TBracket(left_term, list, null);
    }

    /*
    target_list     ::=  target ("," target)* [","]
    target          ::=  identifier
                            | "(" target_list ")"
                            | "[" [target_list] "]"
                            | attributeref
                            | subscription
                            | slicing
                            | "*" target
    */
    TargetList() {
        var list = new Array();

        while (true) {
            list.push( this.Primary() );

            if (this.CurrentToken.Kind != EKind.Comma) {

                break;
            }
            this.GetToken(EKind.Comma);
        }

        return new TTargetList(list);
    }

    // comp_for ::= "for" target_list "in" or_test [comp_iter]
    // comp_iter     ::=  comp_for | comp_if
    CompFor(element_value) {
        this.GetToken(EKind.for_);
        var target_list = this.TargetList();
        this.GetToken(EKind.in_);
        var source = this.OrTest();

        var comp_for = new TCompFor(target_list, source);
        comp_for.ElementValue = element_value;

        if (this.CurrentToken.Kind == EKind.for_) {

            comp_for.InnerFor = this.CompFor(element_value);
        }

        return comp_for;
    }

    SetDictionary() {
        var list = new Array();

        this.GetToken(EKind.LC);

        while (true) {
            var key = this.Expression();
            this.GetToken(EKind.Colon);
            var val = this.Expression();

            list.push(new TKeyValue(key, val));

            if (this.CurrentToken.Kind != EKind.Comma) {

                break;
            }

            this.GetToken(EKind.Comma);
        }

        this.GetToken(EKind.RC);

        return new TSetDictionary(list);
    }

    // generator_expression ::=  "(" expression comp_for ")"
    // argument_list        ::=  positional_arguments ["," starred_and_keywords]
    ExpressionListGenerator() {
        var list = new Array();

        while (true) {
            var trm1 = this.Expression();
            if (this.CurrentToken.Kind == EKind.Assign) {

                this.GetToken(EKind.Assign);
                trm1.AssignedValue = this.Expression();
            }

            if (this.CurrentToken.Kind == EKind.for_) {

                var comp_for = this.CompFor(trm1);
                return comp_for;
            }
            else {

                list.push(trm1);

                if (this.CurrentToken.Kind != EKind.Comma) {

                    break;
                }

                this.GetToken(EKind.Comma);

                if (this.CurrentToken.Kind == EKind.RP) {

                    break;
                }
            }
        }

        return new TExpressionListGenerator(list);
    }

    // parenth_form ::=  "(" [starred_expression] ")"
    //     starred_expression ::=  expression | ( starred_item "," )* [starred_item]
    // generator_expression ::=  "(" expression comp_for ")"
    ParenthesizedFormGenerator() {
        var is_tuple = false;
        var list = new Array();
        var comp_for = null;

        this.GetToken(EKind.LP);
        while (this.CurrentToken.Kind != EKind.RP) {

            var trm1 = this.Expression();
            if (this.CurrentToken.Kind == EKind.Assign) {

                this.GetToken(EKind.Assign);
                trm1.AssignedValue = this.Expression();
            }

            if (this.CurrentToken.Kind == EKind.for_) {

                comp_for = this.CompFor(trm1);
                break;
            }

            list.push(trm1);

            if (this.CurrentToken.Kind != EKind.Comma) {
                break;
            }
            this.GetToken(EKind.Comma);
            if (this.CurrentToken.Kind == EKind.RP) {
                is_tuple = true;
                break;
            }
        }
        this.GetToken(EKind.RP);

        return new TParenthesizedFormGenerator(list, is_tuple, comp_for);
    }

    // primary ::=  atom | attributeref | subscription | slicing | call
    //     atom ::=  identifier | literal | enclosure
    //     attributeref ::=  primary "." identifier
    //     call ::=  primary "(" [argument_list [","] | comprehension] ")"
    Primary() {
        var trm1;

        switch (this.CurrentToken.Kind) {
            case EKind.NumberLiteral:
                trm1 = new TLiteral(this.GetToken(EKind.NumberLiteral));
                break;

            case EKind.StringLiteral:
                trm1 = new TLiteral(this.GetToken(EKind.StringLiteral));
                break;

            case EKind.Identifier:
                trm1 = new TReference(this.GetToken(EKind.Identifier));
                break;

            case EKind.LP:
                trm1 = this.ParenthesizedFormGenerator();
                break;

            case EKind.LB:
                trm1 = this.SubscriptionSlicingCompFor(null);
                break;

            case EKind.LC:
                trm1 = this.SetDictionary();
                break;

            default:
                throw new TParseException(this);
        }

        while(true){

            switch(this.CurrentToken.Kind){
                case EKind.Dot:
                    this.GetToken(EKind.Dot);
                    var id1 = this.GetToken(EKind.Identifier);
                    trm1 = new TDot(trm1, id1);
                    break;

                case EKind.LB:
                    trm1 = this.SubscriptionSlicingCompFor(trm1);
                    break;

                case EKind.LP:
                    trm1 = new TApply(trm1, this.ParenthesizedFormGenerator());
                    break;

                default:
                    return trm1;
            }
        }

        return trm1;
    }

    // power ::=  ( await_expr | primary ) ["**" u_expr]
    PowerExpression() {
        var trm1 = this.Primary();

        if (this.CurrentToken.Kind == EKind.Power) {

            var op_tkn = this.GetToken(EKind.Power);
            trm1 = this.MakeOperation2(op_tkn, trm1, this.UnaryExpression());
        }

        return trm1;
    }

    // u_expr ::=  power | "-" u_expr | "+" u_expr | "~" u_expr
    UnaryExpression() {
        switch (this.CurrentToken.Kind) {
            case EKind.Sub:
            case EKind.Add:
            case EKind.Tilde:
                var op_tkn = this.GetToken(EKind.Undefined);
                
                return this.MakeOperation1(op_tkn, this.UnaryExpression());

            default:
                return this.PowerExpression();
        }
    }

    // m_expr ::=  u_expr | m_expr "*" u_expr | m_expr "@" m_expr | m_expr "//" u_expr| m_expr "/" u_expr | m_expr "%" u_expr
    MultiplicativeExpression() {
        var trm1 = this.UnaryExpression();

        while (true) {
            switch (this.CurrentToken.Kind) {
                case EKind.Mul:
                case EKind.At:
                case EKind.FloorDiv:
                case EKind.Div:
                case EKind.Mod_:
                    var op_tkn = this.GetToken(EKind.Undefined);
                    trm1 = this.MakeOperation2(op_tkn, trm1, this.UnaryExpression());
                    break;

                default:
                    return trm1;
            }
        }
    }

    // a_expr ::=  m_expr | a_expr "+" m_expr | a_expr "-" m_expr
    AdditiveExpression() {
        var trm1 = this.MultiplicativeExpression();

        while (true) {
            switch (this.CurrentToken.Kind) {
                case EKind.Add:
                case EKind.Sub:
                    var op_tkn = this.GetToken(EKind.Undefined);
                    trm1 = this.MakeOperation2(op_tkn, trm1, this.MultiplicativeExpression());
                    break;

                default:
                    return trm1;
            }
        }
    }

    // shift_expr ::=  a_expr | shift_expr ( "<<" | ">>" ) a_expr
    ShiftExpression() {
        var trm1 = this.AdditiveExpression();

        while (true) {
            switch (this.CurrentToken.Kind) {
                case EKind.LeftShiftEq:
                case EKind.RightShiftEq:
                    var op_tkn = this.GetToken(EKind.Undefined);
                    trm1 = this.MakeOperation2(op_tkn, trm1, this.AdditiveExpression());
                    break;

                default:
                    return trm1;
            }
        }
    }

    // and_expr ::=  shift_expr | and_expr "&" shift_expr
    AndExpression() {
        var trm1 = this.ShiftExpression();

        while (this.CurrentToken.Kind == EKind.Anp) {
            var op_tkn = this.GetToken(EKind.Anp);
            trm1 = this.MakeOperation2(op_tkn, trm1, this.ShiftExpression());
        }

        return trm1;
    }

    // xor_expr ::=  and_expr | xor_expr "^" and_expr
    XorExpression() {
        var trm1 = this.AndExpression();

        while (this.CurrentToken.Kind == EKind.Hat) {
            var op_tkn = this.GetToken(EKind.Hat);
            trm1 = this.MakeOperation2(op_tkn, trm1, this.AndExpression());
        }

        return trm1;
    }

    // or_expr  ::=  xor_expr | or_expr "|" xor_expr
    OrExpression() {
        var trm1 = this.XorExpression();

        while (this.CurrentToken.Kind == EKind.BitOR) {
            var op_tkn = this.GetToken(EKind.BitOR);
            trm1 = this.MakeOperation2(op_tkn, trm1, this.XorExpression());
        }

        return trm1;
    }

    // comparison    ::=  or_expr ( comp_operator or_expr )*
    //     comp_operator ::=  "<" | ">" | "==" | ">=" | "<=" | "!=" | "is" ["not"] | ["not"] "in"
    Comparison() {
        var op_tkn;
        var trm1 = this.OrExpression();

        while (true) {
            switch (this.CurrentToken.Kind) {
                case EKind.LT:
                case EKind.GT:
                case EKind.Eq:
                case EKind.GE:
                case EKind.LE:
                case EKind.NE:
                    op_tkn = this.GetToken(EKind.Undefined);
                    trm1 = this.MakeOperation2(op_tkn, trm1, this.OrExpression());
                    break;

                case EKind.is_:
                    this.GetToken(EKind.is_);
                    if(this.CurrentToken.Kind == EKind.not_){

                        this.GetToken(EKind.not_);
                    }
                    this.OrExpression();
                    break;

                case EKind.not_:
                    this.GetToken(EKind.not_);
                    this.GetToken(EKind.in_);
                    this.OrExpression();
                    break;

                case EKind.in_:
                    op_tkn = this.GetToken(EKind.in_);
                    trm1 = this.MakeOperation2(op_tkn, trm1, this.OrExpression());
                    break;

                default:
                    return trm1;
            }
        }
    }

    // not_test ::=  comparison | "not" not_test
    NotTest() {
        if(this.CurrentToken.Kind == EKind.not_){
            var op_tkn = this.GetToken(EKind.not_);
            return this.MakeOperation1(op_tkn, this.NotTest());
        }
        else{
            return this.Comparison();
        }
    }

    // and_test ::=  not_test | and_test "and" not_test
    AndTest() {
        var trm1 = this.NotTest();

        while (this.CurrentToken.Kind == EKind.and_) {
            var op_tkn = this.GetToken(EKind.and_);
            trm1 = this.MakeOperation2(op_tkn, trm1, this.NotTest());
        }

        return trm1;
    }

    // or_test  ::=  and_test | or_test "or" and_test
    OrTest() {
        var trm1 = this.AndTest();

        while (this.CurrentToken.Kind == EKind.or_) {
            var op_tkn = this.GetToken(EKind.or_);
            trm1 = this.MakeOperation2(op_tkn, trm1, this.AndTest());
        }

        return trm1;
    }

    // conditional_expression ::=  or_test ["if" or_test "else" expression]
    ConditionalExpression() {
        return this.OrTest();
    }

    // expression ::=  conditional_expression | lambda_expr
    Expression() {
        return this.ConditionalExpression();
    }

    // stmt_list ::=  simple_stmt (";" simple_stmt)* [";"]
    StatementList() {
        var list = new Array();

        while(this.CurrentToken.Kind != EKind.EOT){
            list.push( this.SimpleStatement() );
            if (this.CurrentToken.Kind != EKind.EOT) {

                this.GetToken(EKind.SemiColon);
            }
        }

        return list;
    }

    ReadClass() {
        this.GetToken(EKind.class_);
        var id1 = this.GetToken(EKind.Identifier);
        if (this.CurrentToken.Kind == EKind.LP) {

            this.GetToken(EKind.LP);
            while (true) {
                var id2 = this.GetToken(EKind.Identifier);

                if (this.CurrentToken.Kind != EKind.Comma) {
                    break;
                }
                this.GetToken(EKind.Comma);
            }
            this.GetToken(EKind.RP);
        }

        this.GetToken(EKind.Colon);
        this.GetToken(EKind.EOT);

        return new TClass(id1.TextTkn);
    }

    ReadDef() {
        this.GetToken(EKind.def_);
        var id1 = this.GetToken(EKind.Identifier);

        this.GetToken(EKind.LP);
        var params = new Array();
        while (true) {
            var var1 = new TVariable( this.GetToken(EKind.Identifier) );

            if (this.CurrentToken.Kind == EKind.Assign) {
                this.GetToken(EKind.Assign);
                var1.DefaultValue = this.Expression();
            }

            params.push(var1);

            if (this.CurrentToken.Kind != EKind.Comma) {
                break;
            }
            this.GetToken(EKind.Comma);
        }
        this.GetToken(EKind.RP);

        this.GetToken(EKind.Colon);

        if (this.CurrentToken.Kind != EKind.EOT) {

            this.StatementList();
        }

        this.GetToken(EKind.EOT);

        return new TFunction(id1.TextTkn, params);
    }

    StatementEnd() {
        var list = null;

        this.GetToken(EKind.Colon);

        if (this.CurrentToken.Kind != EKind.EOT) {

            list = this.StatementList();
        }

        this.GetToken(EKind.EOT);

        return list;
    }

    // augop ::=  "+=" | "-=" | "*=" | "@=" | "/=" | "//=" | "%=" | "**=" | ">>=" | "<<=" | "&=" | "^=" | "|="
    ReadAssignment() {
        var target_list = this.TargetList();
        Assert(target_list != null, "Read Assignment Target List");

        switch (this.CurrentToken.Kind) {
            case EKind.Assign:
            case EKind.PowerEq:
            case EKind.AddEq:
            case EKind.SubEq:
            case EKind.MulEq:
            case EKind.DivEq:
            case EKind.ModEq :
            case EKind.LeftShiftEq:
            case EKind.RightShiftEq:
                this.GetToken(EKind.Undefined);
                var val = this.ExpressionListGenerator();
                Assert(val != null, "Read Assignment Value");
                return new TAssignment(target_list, val);

            default:
                throw new TParseException(this);
        }
    }

    ReadCall() {
        var trm1 = this.ExpressionListGenerator();
        Assert(trm1 != null, "Read Call");

        return new TCall(trm1);
    }

    ReadAssignmentCall(){
        var open_cnt = 0;
        var is_assign = false;

        for_loop: for(let tkn of this.TokenList){
            switch(tkn.Kind){
                case EKind.LP:
                case EKind.LB:
                case EKind.LC:
                    open_cnt++;
                    break;
        
                case EKind.RP:
                case EKind.RB:
                case EKind.RC :
                    open_cnt--;
                    break;

                case EKind.Assign:
                case EKind.PowerEq:
                case EKind.AddEq:
                case EKind.SubEq:
                case EKind.MulEq:
                case EKind.DivEq:
                case EKind.ModEq:
                case EKind.LeftShiftEq:
                case EKind.RightShiftEq:
                    if(open_cnt == 0){

                        is_assign = true;
                        break for_loop;
                    }
                    break;

            }
        }

        if(is_assign){

            return this.ReadAssignment();
        }
        else{

            return this.ReadCall();
        }
    }

    IdentifierList() {
        var id1 = this.GetToken(EKind.Identifier);
        while (this.CurrentToken.Kind == EKind.Dot) {

            this.GetToken(EKind.Dot);
            this.GetToken(EKind.Identifier);
        }
    }

    ReadImport() {
        if (this.CurrentToken.Kind == EKind.from_) {

            this.GetToken(EKind.from_);
            this.IdentifierList();
        }

        this.GetToken(EKind.import_);
        this.IdentifierList();

        if (this.CurrentToken.Kind == EKind.as_) {

            this.GetToken(EKind.as_);
            var id2 = this.GetToken(EKind.Identifier);
        }
        this.GetToken(EKind.EOT);

        return new TImport();
    }

    ReadReturn() {
        var trm1 = null;

        this.GetToken(EKind.return_);
        if (this.CurrentToken.Kind != EKind.EOT) {
            trm1 = this.ExpressionListGenerator();
            Assert(trm1 != null, "Read Return");
        }

        return new TReturn(trm1);
    }

    ReadPrint() {
        var trm1 = null;
        this.GetToken(EKind.print_);

        if(this.CurrentToken.Kind != EKind.EOT){

            var trm1 = this.ExpressionListGenerator();

            //var trm1 = this.Expression();
            //Assert(trm1 != null, "Read Print 1");

            //while (this.CurrentToken.Kind == EKind.Comma) {

            //    this.GetToken(EKind.Comma);
            //    trm1 = this.Expression();
            //    Assert(trm1 != null, "Read Print 2");
            //}
        }

        return new TPrint(trm1);
    }

    ReadPass() {
        this.GetToken(EKind.pass_);

        return new TPass();
    }

    SimpleStatement() {
        switch(this.CurrentToken.Kind){
            case EKind.return_:
                return this.ReadReturn();

            case EKind.print_:
                return this.ReadPrint();

            case EKind.pass_:
                return this.ReadPass();

            default:
                return this.ReadAssignmentCall();
        }
    }

    // for_stmt ::=  "for" target_list "in" expression_list ":" suite
    ReadFor() {
        var comp_for = this.CompFor(null);
        Assert(comp_for != null, "Read For");

        return new TFor(comp_for, this.StatementEnd());
    }

    // if_stmt ::=  "if" expression ":" suite
    ReadIf() {
        this.GetToken(EKind.if_);
        var trm1 = this.Expression();
        Assert(trm1 != null, "Read If");

        return new TIf(trm1, this.StatementEnd());
    }

    ReadElse() {
        this.GetToken(EKind.else_);
        this.StatementEnd();

        return new TElse();
    }

    // try1_stmt ::=  "try" ":" suite
    ReadTry() {
        this.GetToken(EKind.try_);
        this.StatementEnd();

        return new TTry();
    }

    // "except" [expression ["as" identifier]] ":" suite
    ReadExcept() {
        this.GetToken(EKind.except_);
        if(this.CurrentToken.Kind != EKind.Colon){

            var trm1 = this.Expression();
            Assert(trm1 != null, "Read Except");

            if (this.CurrentToken.Kind == EKind.as_) {

                this.GetToken(EKind.as_);
                var id1 = this.GetToken(EKind.Identifier);
            }
        }

        this.StatementEnd();

        return new TExcept();
    }

    ParseLines(lines) {
        var src = new TSource();
        src.Lines = new Array();

        var last_token_type = ETokenType.Undefined;
        for (var i in lines) {
            //console.log(lines[i]);
            var text = lines[i];
            var tokens = Lexer.LexicalAnalysis(text, last_token_type);
            for (let tkn of tokens) {

            //console.log(EnumName(ETokenType, tkn.TokenType) + " " + EnumName(EKind, tkn.Kind) + " " + tkn.TextTkn);
                last_token_type = tkn.TokenType;
            }

            src.Lines.push(new TLine(text, tokens));
        }

        return src;
    }

    ParseStatement(tokens, line_text) {
        var stmt = null;

        this.TokenList = tokens;
        this.TokenPos = 0;
        this.LineText = line_text;

        this.CurrentToken = this.TokenList[0];

        if (1 < this.TokenList.length) {

            this.NextToken = this.TokenList[1];
        }
        else {

            this.NextToken = this.EOTToken;
        }

        switch (this.CurrentToken.Kind) {
            case EKind.class_:
                return this.ReadClass();

            case EKind.def_:
                return this.ReadDef();

            case EKind.elif_:
                console.log("elif_ " + this.LineText);
                return null;

            case EKind.else_:
                return this.ReadElse();

            case EKind.for_:
                return this.ReadFor();

            case EKind.if_:
                return this.ReadIf();

            case EKind.try_:
                return this.ReadTry();

            case EKind.except_:
                return this.ReadExcept();

            case EKind.finally_:
                console.log("finally_ " + this.LineText);
                return null;

            case EKind.from_:
            case EKind.import_:
                return this.ReadImport();

            case EKind.return_:
                stmt = this.ReadReturn();
                this.GetToken(EKind.EOT);
                return stmt;

            case EKind.print_:
                stmt = this.ReadPrint();
                this.GetToken(EKind.EOT);
                return stmt;

            case EKind.pass_:
                stmt = this.ReadPass();
                this.GetToken(EKind.EOT);
                return stmt;

            case EKind.At:
                this.GetToken(EKind.At);
                this.GetToken(EKind.Identifier);
                this.GetToken(EKind.EOT);
                //console.log("Decorator " + this.LineText);
                return new TDecorator();

            case EKind.Identifier:
            default:
                stmt = this.ReadAssignmentCall();
                this.GetToken(EKind.EOT);
                return stmt;
        }

        return stmt;
    }

    ParseAllStatement(src) {
        var tokens = new Array();
        var open_cnt = 0;
        var start_line = null;
        var start_line_idx = 0;
        var line_text = "";
        var obj_stack = new Array();

        for (var lidx = 0; lidx < src.Lines.length; lidx++) {
            var line = src.Lines[lidx];

            if (line.Tokens.length != 0) {
                // 空行でない場合

                if (start_line == null) {
                    start_line = line;
                    start_line_idx = lidx;
                    line_text = line.Text;
                }
                else {

                    line_text += "\r\n" + line.Text;
                }

                tokens = tokens.concat(line.Tokens);

                for(let tkn of line.Tokens) {
                    switch (tkn.Kind) {
                        case EKind.LP:
                        case EKind.LB:
                        case EKind.LC:
                            open_cnt++;
                            break;
                        case EKind.RP:
                        case EKind.RB:
                        case EKind.RC:
                            open_cnt--;
                            break;
                    }
                }

                if (tokens[tokens.length - 1].Kind == EKind.Backslash) {
                    // 末尾が\の場合

                    tokens.pop();
                }
                else if (open_cnt == 0) {
                    // 継続行でない場合

                    var top_kind = tokens[0].Kind;
                    if(top_kind != EKind.LineComment && top_kind != EKind.StringLiteral){

                        this.LineIdx = start_line_idx;
                        var obj = this.ParseStatement(tokens, line_text)
                        start_line.ObjLine = obj;

                        if (!obj) {
                            console.log("null statement:" + line_text);
                        }

                        while (obj_stack.length != 0 && tokens[0].StartPos <= obj_stack[ obj_stack.length - 1].Indent) {
                            obj_stack.pop();
                        }

                        if (obj_stack.length == 0) {

                            src.CodeList.push(obj);
                        }
                        else {

                            var parent_obj = obj_stack[obj_stack.length - 1].Obj;
                            obj.Parent = parent_obj;
                            if (parent_obj instanceof TStatement) {

                                Assert(parent_obj.Statements, "Illegal parent statement:(" + start_line_idx + ") " + line_text);
                                Assert(obj instanceof TStatement, "Illegal Statement child:(" + start_line_idx + ") " + line_text);

                                parent_obj.Statements.push(obj);
                            }
                            else if (parent_obj instanceof TFunction) {

                                if(obj instanceof TStatement){

                                    parent_obj.Statements.push(obj);
                                }
                                else if(obj instanceof TFunction){

                                    parent_obj.InnerFunctions.push(obj);
                                }
                                else{

                                    Assert(false, "Illegal function child:(" +start_line_idx + ") " +line_text);
                                }
                            }
                            else if (parent_obj instanceof TClass) {

                                if (obj instanceof TFunction) {

                                    parent_obj.Functions.push(obj);
                                }
                                else if (obj instanceof TDecorator) {

                                }
                                else {

                                    Assert(false, "Illegal class member:(" + start_line_idx + ") " + line_text);
                                }
                            }
                            else {
                                
                                Assert(false, "Illegal Statement");
                            }
                        }

                        obj_stack.push({ Indent: tokens[0].StartPos, Obj: obj } );
                    }

                    tokens = new Array();
                    start_line = null;
                    line_text = "";
                }
            }
        }
    }

    Parse(txt) {
        txt = txt.replace(/\r\n/g, "\n");
        var lines = txt.split("\n");
        var src = this.ParseLines(lines);

        this.ParseAllStatement(src);

        return src;
    }

    /*
     * 指定された種類の字句を読む。
     */
    GetToken(type) {

        if (type != EKind.Undefined && type != this.CurrentToken.Kind) {

            throw new TParseException(this);
        }

        var tkn = this.CurrentToken;

        while_block: while (true) {

            this.TokenPos++;
            if (this.TokenPos < this.TokenList.length) {

                this.CurrentToken = this.TokenList[this.TokenPos];

                switch (this.CurrentToken.TokenType) {
                    case ETokenType.LineComment:
                        break;

                    default:
                        break while_block;
                }
            }
            else {

                this.CurrentToken = this.EOTToken;
                break;
            }
        }

        if (this.TokenPos + 1 < this.TokenList.length) {

            this.NextToken = this.TokenList[this.TokenPos + 1];
        }
        else {

            this.NextToken = this.EOTToken;
        }

        return tkn;
    }
}
