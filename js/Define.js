function EnumName(e, i) {
    for (name in e) {
        if (e[name] == i) {
            return name;
        }
    }

    return "unknown";
}

/*
    字句型
*/
const ETokenType = {
    // 未定義
    Undefined: Symbol(),

    // 空白
    EOT: Symbol(),

    // 文字
    Char_: Symbol(),

    // 文字列
    String_: Symbol(),

    // 逐語的文字列 ( @"文字列" )
    VerbatimString: Symbol("VerbatimString"),

    // "で閉じてない逐語的文字列
    VerbatimStringContinued: Symbol("VerbatimStringContinued"),

    // 識別子
    Identifier: Symbol(),

    // キーワード
    Keyword: Symbol(),
    Int: Symbol(),
    Float: Symbol(),
    Double: Symbol(),

    // 記号
    Symbol: Symbol(),

    // 行コメント      ( // )
    LineComment: Symbol(),

    // エラー
    Error: Symbol()
}

const EKind = {
    Undefined: Symbol(),

    Identifier: Symbol(),
    ClassName: Symbol(),
    NumberLiteral: Symbol(),
    StringLiteral: Symbol(),

    NewInstance: Symbol(),
    NewArray: Symbol(),

    Index: Symbol(),

    FunctionApply: Symbol(),

    // 行コメント      ( // )
    LineComment: Symbol(),

    NL: Symbol(),
    Tab: Symbol(),
    Space: Symbol(),
    EOL: Symbol(),

    LP: Symbol(),
    RP: Symbol(),
    LB: Symbol(),
    RB: Symbol(),
    LC: Symbol(),
    RC: Symbol(),
    Dot: Symbol(),
    Power: Symbol(),
    Dec: Symbol(),
    Add: Symbol(),
    Sub: Symbol(),
    Mul: Symbol(),
    Div: Symbol(),
    FloorDiv: Symbol(),
    Mod_: Symbol(),
    At: Symbol(),
    LeftShift: Symbol(),
    RightShift: Symbol(),

    Assign: Symbol(),
    PowerEq: Symbol(),
    AddEq: Symbol(),
    SubEq: Symbol(),
    MulEq: Symbol(),
    DivEq: Symbol(),
    ModEq: Symbol(),
    AtEq: Symbol(),
    LeftShiftEq: Symbol(),
    RightShiftEq: Symbol(),
    Anp: Symbol(),
    Comma: Symbol(),
    Colon: Symbol(),
    SemiColon: Symbol(),
    Question: Symbol(),
    Hat: Symbol(),
    BitOR: Symbol(),
    Tilde: Symbol(),
    Eq: Symbol(),
    NE: Symbol(),
    LT: Symbol(),
    GT: Symbol(),
    LE: Symbol(),
    GE: Symbol(),
    Or_: Symbol(),
    And_: Symbol(),
    Not_: Symbol(),
    Lambda: Symbol(),
    Backslash: Symbol(),

    and_: Symbol(),
    as_: Symbol(),
    assert_: Symbol(),
    break_: Symbol(),
    class_: Symbol(),
    continue_: Symbol(),
    def_: Symbol(),
    del_: Symbol(),
    elif_: Symbol(),
    else_: Symbol(),
    except_: Symbol(),
    finally_: Symbol(),
    for_: Symbol(),
    from_: Symbol(),
    global_: Symbol(),
    if_: Symbol(),
    import_: Symbol(),
    in_: Symbol(),
    is_: Symbol(),
    lambda_: Symbol(),
    nonlocal_: Symbol(),
    not_: Symbol(),
    or_: Symbol(),
    pass_: Symbol(),
    print_: Symbol(),
    raise_: Symbol(),
    return_: Symbol(),
    try_: Symbol(),
    while_: Symbol(),
    with_: Symbol(),
    yield_: Symbol(),

    EOT: Symbol()
}
