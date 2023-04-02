from piss import lex
from piss import parse
from piss.lex import KeywordKind, Span
from piss.parse import Parser
from textwrap import dedent


def test_parse_single_token() -> None:
    input = "{"
    expected_token = lex.Token(lex.LeftBrace(), span=Span(0, 1))

    tokens = lex.tokenize(input)
    parser = Parser(tokens)

    assert parser.parse_token(lex.LeftBrace) == expected_token


def test_parse_second_token() -> None:
    input = "{}"

    expected_token = lex.Token(lex.RightBrace(), span=Span(1, 2))

    tokens = lex.tokenize(input)
    parser = Parser(tokens)

    parser.parse_token(lex.LeftBrace)

    assert parser.parse_token(lex.RightBrace) == expected_token


def test_parse_keyword() -> None:
    input = "struct"

    tokens = lex.tokenize(input)
    parser = Parser(tokens)

    assert parser.parse_keyword(KeywordKind.Struct) == parse.Keyword(
        Span(0, 6), KeywordKind.Struct
    )


def test_parse_second_keyword() -> None:
    tokens = lex.tokenize("; module")

    parser = Parser(tokens)
    parser.parse_token(lex.SemiColon)

    assert parser.parse_keyword(KeywordKind.Module) == parse.Keyword(
        Span(2, 8),
        KeywordKind.Module,
    )


def test_parse_type_int() -> None:
    uint = Parser(lex.tokenize("int")).parse_type()

    assert isinstance(uint, parse.PrimitiveType)
    assert uint.type == parse.PrimitiveKind.Int


def test_parse_type_uint() -> None:
    uint = Parser(lex.tokenize("uint")).parse_type()

    assert isinstance(uint, parse.PrimitiveType)
    assert uint.type == parse.PrimitiveKind.Uint


def test_parse_field_int() -> None:
    tokens = lex.tokenize("int MyInt")
    field = Parser(tokens).parse_field()

    assert field.ident.name == "MyInt"
    assert isinstance(field.kind, parse.PrimitiveType)
    assert field.kind.type == parse.PrimitiveKind.Int


def test_parse_field_uint() -> None:
    tokens = lex.tokenize("uint MyUint")
    field = Parser(tokens).parse_field()

    assert field.ident.name == "MyUint"
    assert isinstance(field.kind, parse.PrimitiveType)
    assert field.kind.type == parse.PrimitiveKind.Uint


def test_parse_const_primitive_integer() -> None:
    tokens = lex.tokenize("const int TRIANGLE_SIDES = 3;")
    const = Parser(tokens).parse_const()

    # For now, ignore Span checks
    assert const.ident.name == "TRIANGLE_SIDES"

    assert isinstance(const.kind, parse.PrimitiveType)
    assert const.kind.type == parse.PrimitiveKind.Int

    assert isinstance(const.expr.expr, parse.Integer)
    assert const.expr.expr.value == 3


def test_parse_const_primitive_identifier() -> None:
    tokens = lex.tokenize("const int TRIANGLE_SIDES = THREE;")
    const = Parser(tokens).parse_const()

    # For now, ignore Span checks
    assert const.ident.name == "TRIANGLE_SIDES"

    assert isinstance(const.kind, parse.PrimitiveType)
    assert const.kind.type == parse.PrimitiveKind.Int

    assert isinstance(const.expr.expr, parse.Identifier)
    assert const.expr.expr.name == "THREE"


def test_parse_const_identifier_integer() -> None:
    tokens = lex.tokenize("const MyInt MyValue = 3;")
    const = Parser(tokens).parse_const()

    # For now, ignore Span checks
    assert const.ident.name == "MyValue"

    assert isinstance(const.kind, parse.IdentifierType)
    assert const.kind.type.name == "MyInt"

    assert isinstance(const.expr.expr, parse.Integer)
    assert const.expr.expr.value == 3


def test_parse_const_identifier_identifier() -> None:
    tokens = lex.tokenize("const MyInt MyValue = SOME_ALIAS;")
    const = Parser(tokens).parse_const()

    # For now, ignore Span checks
    assert const.ident.name == "MyValue"

    assert isinstance(const.kind, parse.IdentifierType)
    assert const.kind.type.name == "MyInt"

    assert isinstance(const.expr.expr, parse.Identifier)
    assert const.expr.expr.name == "SOME_ALIAS"


def test_parse_struct() -> None:
    input = dedent(
        """
        struct Token {
            TokenKind kind,
            Span span,
        };
        """
    )

    tokens = lex.tokenize(input)

    parser = Parser(tokens)

    struct = parser.parse_struct()

    # For now, ignore Span checks.
    assert struct.ident.name == "Token"
    assert len(struct.fields) == 2

    assert struct.fields[0].ident.name == "kind"

    assert isinstance(struct.fields[0].kind, parse.IdentifierType)
    assert struct.fields[0].kind.type.name == "TokenKind"

    assert struct.fields[1].ident.name == "span"

    assert isinstance(struct.fields[1].kind, parse.IdentifierType)
    assert struct.fields[1].kind.type.name == "Span"


def test_parse_struct_primitive_field() -> None:
    input = dedent(
        """
        struct Person {
            int age,
            uint wealth,
        };
        """
    )

    tokens = lex.tokenize(input)
    struct = Parser(tokens).parse_struct()

    # For now, ignore Span checks
    assert struct.ident.name == "Person"


def test_parse_empty_struct() -> None:
    tokens = lex.tokenize("struct Empty {};")
    struct = Parser(tokens).parse_struct()

    # For now, ignore Span checks
    assert struct.ident.name == "Empty"
    assert len(struct.fields) == 0


def test_parse_enum() -> None:
    input = dedent(
        """
        enum Bool {
            True,
            False,
        };
        """
    )

    tokens = lex.tokenize(input)
    enum = Parser(tokens).parse_enum()

    # For now, ignore Span checks
    assert enum.ident.name == "Bool"
    assert len(enum.variants) == 2
    assert enum.variants[0].name == "True"
    assert enum.variants[1].name == "False"


def test_parse_empty_enum() -> None:
    tokens = lex.tokenize("enum Unit {};")
    enum = Parser(tokens).parse_enum()

    # For now, ignore Span checks
    assert enum.ident.name == "Unit"
    assert len(enum.variants) == 0
