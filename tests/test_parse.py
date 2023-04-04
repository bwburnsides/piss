"""
Tests for functions in parse.py
"""

import pytest
from typing import Callable, NamedTuple
from functools import partial

from piss import lex, node, parse
from piss.parse import Parser
from piss.lex import KeywordKind, Span

SimpleToken: Callable[[lex.TokenKind], lex.Token[lex.TokenKind]] = partial(
    lex.Token[lex.TokenKind], span=Span()
)


class SimpleFieldType(NamedTuple):
    type: lex.Identifier
    name: lex.Identifier
    node: node.Field


class PrimitiveTypedefType(NamedTuple):
    tokens: list[lex.Token[lex.TokenKind]]
    node: node.Typedef


class MultiVariantEnumType(NamedTuple):
    tokens: list[lex.Token[lex.TokenKind]]
    node: node.Enum


class MultiFieldStructType(NamedTuple):
    tokens: list[lex.Token[lex.TokenKind]]
    node: node.Struct


SimpleFieldFixtureType: Callable[
    [Callable[[], SimpleFieldType]],
    Callable[[], SimpleFieldType],
] = pytest.fixture


@SimpleFieldFixtureType
def simple_field_foo() -> SimpleFieldType:
    type = SimpleToken(lex.Identifier("FooType"))
    assert isinstance(type.kind, lex.Identifier)

    ident = SimpleToken(lex.Identifier("foo"))
    assert isinstance(ident.kind, lex.Identifier)

    expected_node = node.Field(
        span=Span(),
        kind=node.IdentifierType(
            type.span,
            node.Identifier(type.span, type.kind.name),
        ),
        ident=node.Identifier(ident.span, ident.kind.name),
    )

    return SimpleFieldType(
        type.kind,
        ident.kind,
        expected_node,
    )


@SimpleFieldFixtureType
def simple_field_bar() -> SimpleFieldType:
    type = SimpleToken(lex.Identifier("BarType"))
    assert isinstance(type.kind, lex.Identifier)

    ident = SimpleToken(lex.Identifier("bar"))
    assert isinstance(ident.kind, lex.Identifier)

    expected_node = node.Field(
        span=Span(),
        kind=node.IdentifierType(
            type.span,
            node.Identifier(type.span, type.kind.name),
        ),
        ident=node.Identifier(ident.span, ident.kind.name),
    )

    return SimpleFieldType(
        type.kind,
        ident.kind,
        expected_node,
    )


PrimitiveTypedefFixtureType: Callable[
    [Callable[[], PrimitiveTypedefType]],
    Callable[[], PrimitiveTypedefType],
] = pytest.fixture


@PrimitiveTypedefFixtureType
def primitive_typedef() -> PrimitiveTypedefType:
    typedef = node.Typedef(
        Span(),
        node.Identifier(Span(), "FooType"),
        node.PrimitiveType(Span(), node.PrimitiveKind.Int),
    )

    tokens = [
        SimpleToken(lex.Keyword(KeywordKind.Typedef)),
        SimpleToken(lex.Keyword(KeywordKind.Int)),
        SimpleToken(lex.Identifier("FooType")),
        SimpleToken(lex.SemiColon()),
    ]

    return PrimitiveTypedefType(tokens, typedef)


MultiVariantEnumFixtureType: Callable[
    [Callable[[], MultiVariantEnumType]],
    Callable[[], MultiVariantEnumType],
] = pytest.fixture


@MultiVariantEnumFixtureType
def multi_variant_enum() -> MultiVariantEnumType:
    tokens = [
        SimpleToken(lex.Keyword(KeywordKind.Enum)),
        SimpleToken(lex.Identifier("Color")),
        SimpleToken(lex.LeftBrace()),
        SimpleToken(lex.Identifier("RED")),
        SimpleToken(lex.Comma()),
        SimpleToken(lex.Identifier("GREEN")),
        SimpleToken(lex.Comma()),
        SimpleToken(lex.Identifier("BLUE")),
        SimpleToken(lex.Comma()),
        SimpleToken(lex.RightBrace()),
        SimpleToken(lex.SemiColon()),
    ]

    enum = node.Enum(
        Span(),
        node.Identifier(Span(), "Color"),
        [
            node.Identifier(Span(), "RED"),
            node.Identifier(Span(), "GREEN"),
            node.Identifier(Span(), "BLUE"),
        ],
    )

    return MultiVariantEnumType(tokens, enum)


MultiFieldStructFixtureType: Callable[
    [Callable[[], MultiFieldStructType]],
    Callable[[], MultiFieldStructType],
] = pytest.fixture


@MultiFieldStructFixtureType
def multi_field_struct() -> MultiFieldStructType:
    tokens = [
        SimpleToken(lex.Keyword(KeywordKind.Struct)),
        SimpleToken(lex.Identifier("MyStruct")),
        SimpleToken(lex.LeftBrace()),
        SimpleToken(lex.Identifier("TypeA")),
        SimpleToken(lex.Identifier("A")),
        SimpleToken(lex.Comma()),
        SimpleToken(lex.Identifier("TypeB")),
        SimpleToken(lex.Identifier("B")),
        SimpleToken(lex.Comma()),
        SimpleToken(lex.RightBrace()),
        SimpleToken(lex.SemiColon()),
    ]

    type_a = SimpleToken(lex.Identifier("TypeA"))
    assert isinstance(type_a.kind, lex.Identifier)

    ident_a = SimpleToken(lex.Identifier("A"))
    assert isinstance(ident_a.kind, lex.Identifier)

    field_a = node.Field(
        span=Span(),
        kind=node.IdentifierType(
            span=type_a.span,
            type=node.Identifier(type_a.span, type_a.kind.name),
        ),
        ident=node.Identifier(ident_a.span, ident_a.kind.name),
    )

    type_b = SimpleToken(lex.Identifier("TypeB"))
    assert isinstance(type_b.kind, lex.Identifier)

    ident_b = SimpleToken(lex.Identifier("B"))
    assert isinstance(ident_b.kind, lex.Identifier)

    field_b = node.Field(
        span=Span(),
        kind=node.IdentifierType(
            span=type_b.span,
            type=node.Identifier(type_b.span, type_b.kind.name),
        ),
        ident=node.Identifier(ident_b.span, ident_b.kind.name),
    )

    struct = node.Struct(
        Span(),
        node.Identifier(Span(), "MyStruct"),
        [field_a, field_b],
    )

    return MultiFieldStructType(tokens, struct)


def test_parse_single_token() -> None:
    expected_token = SimpleToken(lex.LeftBrace())
    parser = Parser([expected_token])

    token = parser.parse_token(lex.LeftBrace)
    assert token == expected_token


def test_parse_second_token() -> None:
    expected_token = SimpleToken(lex.RightBrace())

    parser = Parser([SimpleToken(lex.LeftBrace()), expected_token])
    parser.parse_token(lex.LeftBrace)

    assert parser.parse_token(lex.RightBrace) == expected_token


def test_parse_keyword() -> None:
    token = SimpleToken(lex.Keyword(KeywordKind.Module))
    expected_node = node.Keyword(token.span, KeywordKind.Module)

    parser = Parser([token])

    assert parser.parse_keyword(KeywordKind.Module) == expected_node


def test_parse_second_keyword() -> None:
    token = SimpleToken(lex.Keyword(KeywordKind.Module))
    expected_node = node.Keyword(token.span, KeywordKind.Module)

    parser = Parser(
        [
            SimpleToken(lex.RightBrace()),
            token,
        ]
    )
    parser.parse_token(lex.RightBrace)

    assert parser.parse_keyword(KeywordKind.Module) == expected_node


def test_parse_identifier() -> None:
    token = SimpleToken(lex.Identifier("FooIdentifier"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = node.Identifier(token.span, token.kind.name)

    parser = Parser([token])
    assert parser.parse_identifier() == expected_node


def test_parse_second_identifier() -> None:
    token = SimpleToken(lex.Identifier("FooIdentifier"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = node.Identifier(token.span, token.kind.name)

    parser = Parser(
        [
            SimpleToken(lex.RightBrace()),
            token,
        ]
    )
    parser.parse_token(lex.RightBrace)

    assert parser.parse_identifier() == expected_node


def test_parse_integer() -> None:
    token = SimpleToken(lex.Integer(5))
    assert isinstance(token.kind, lex.Integer)

    expected_node = node.Integer(token.span, token.kind.value)

    parser = Parser([token])

    assert parser.parse_integer() == expected_node


def test_parse_second_integer() -> None:
    token = SimpleToken(lex.Integer(5))
    assert isinstance(token.kind, lex.Integer)

    expected_node = node.Integer(token.span, token.kind.value)

    parser = Parser(
        [
            SimpleToken(lex.LeftBrace()),
            token,
        ]
    )
    parser.parse_token(lex.LeftBrace)

    assert parser.parse_integer() == expected_node


def test_parse_integer_expression() -> None:
    token = SimpleToken(lex.Integer(5))
    assert isinstance(token.kind, lex.Integer)

    expected_node = node.Expression(
        token.span,
        node.Integer(token.span, token.kind.value),
    )

    parser = Parser([token])

    assert parser.parse_expression() == expected_node


def test_parse_identifier_expression() -> None:
    token = SimpleToken(lex.Identifier("foo"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = node.Expression(
        token.span,
        node.Identifier(token.span, token.kind.name),
    )

    parser = Parser([token])

    assert parser.parse_expression() == expected_node


def test_parse_primitive_type() -> None:
    token = SimpleToken(lex.Keyword(KeywordKind.Int))

    expected_node = node.PrimitiveType(
        token.span,
        node.PrimitiveKind.Int,
    )

    parser = Parser([token])

    assert parser.parse_type() == expected_node


def test_parse_identifier_type() -> None:
    token = SimpleToken(lex.Identifier("FooType"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = node.IdentifierType(
        token.span,
        node.Identifier(token.span, token.kind.name),
    )

    parser = Parser([token])

    assert parser.parse_type() == expected_node


def test_parse_array_type() -> None:
    # int[5]

    expected_node = node.ArrayType(
        Span(),
        node.PrimitiveType(Span(), node.PrimitiveKind.Int),
        node.Expression(Span(), node.Integer(Span(), 5)),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Int)),
            SimpleToken(lex.LeftBracket()),
            SimpleToken(lex.Integer(5)),
            SimpleToken(lex.RightBracket()),
        ]
    )

    assert parser.parse_type() == expected_node


def test_parse_nested_array_type() -> None:
    # int[5][ConstFoo]

    expected_node = node.ArrayType(
        Span(),
        node.ArrayType(
            Span(),
            node.PrimitiveType(Span(), node.PrimitiveKind.Int),
            node.Expression(Span(), node.Integer(Span(), 5)),
        ),
        node.Expression(Span(), node.Identifier(Span(), "ConstFoo")),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Int)),
            SimpleToken(lex.LeftBracket()),
            SimpleToken(lex.Integer(5)),
            SimpleToken(lex.RightBracket()),
            SimpleToken(lex.LeftBracket()),
            SimpleToken(lex.Identifier("ConstFoo")),
            SimpleToken(lex.RightBracket()),
        ]
    )

    assert parser.parse_type() == expected_node


def test_parse_field_with_primitive_type() -> None:
    type = SimpleToken(lex.Keyword(KeywordKind.Int))
    ident = SimpleToken(lex.Identifier("foo"))
    assert isinstance(ident.kind, lex.Identifier)

    expected_node = node.Field(
        span=Span(),
        kind=node.PrimitiveType(type.span, node.PrimitiveKind.Int),
        ident=node.Identifier(ident.span, ident.kind.name),
    )

    parser = Parser([type, ident])

    assert parser.parse_field() == expected_node


def test_parse_field_with_identifier_type(simple_field_foo: SimpleFieldType) -> None:
    type, ident, expected_node = simple_field_foo

    parser = Parser([SimpleToken(type), SimpleToken(ident)])

    assert parser.parse_field() == expected_node


def test_parse_const(simple_field_foo: SimpleFieldType) -> None:
    expr = SimpleToken(lex.Integer(5))
    assert isinstance(expr.kind, lex.Integer)

    # FooType foo = 5;

    expected_node = node.Const(
        Span(),
        simple_field_foo.node.ident,
        simple_field_foo.node.kind,
        node.Expression(
            Span(),
            node.Integer(Span(), expr.kind.value),
        ),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Const)),
            SimpleToken(simple_field_foo.type),
            SimpleToken(simple_field_foo.name),
            SimpleToken(lex.Equals()),
            expr,
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_const() == expected_node


def test_parse_simple_struct(simple_field_foo: SimpleFieldType) -> None:
    struct_ident = node.Identifier(Span(), "MyStruct")

    # struct MyStruct {
    #   FooType foo,
    # };

    expected_node = node.Struct(
        Span(),
        struct_ident,
        [
            simple_field_foo.node,
        ],
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Struct)),
            SimpleToken(lex.Identifier("MyStruct")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.Identifier("FooType")),
            SimpleToken(lex.Identifier("foo")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.RightBrace()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_struct() == expected_node


def test_parse_empty_struct() -> None:
    # struct MyStruct {};

    expected_node = node.Struct(
        Span(),
        node.Identifier(Span(), "MyStruct"),
        [],
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Struct)),
            SimpleToken(lex.Identifier("MyStruct")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.RightBrace()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_struct() == expected_node


def test_parse_two_field_struct(multi_field_struct: MultiFieldStructType) -> None:
    # struct MyStruct {
    #   TypeA A,
    #   TypeB B,
    # };

    parser = Parser(multi_field_struct.tokens)

    assert parser.parse_struct() == multi_field_struct.node


def test_parse_struct_missing_semicolon_at_end_of_stream() -> None:
    # struct MyStruct {
    #   TypeA A,
    #   TypeB B,
    # }

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Struct)),
            SimpleToken(lex.Identifier("MyStruct")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.Identifier("TypeA")),
            SimpleToken(lex.Identifier("A")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.Identifier("TypeB")),
            SimpleToken(lex.Identifier("B")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.RightBrace()),
        ]
    )

    with pytest.raises(parse.UnexpectedEOF):
        parser.parse_struct()


def test_parse_struct_missing_right_brace_in_stream() -> None:
    # struct MyStruct {
    #   TypeA A,
    #   TypeB B,
    # ];

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Struct)),
            SimpleToken(lex.Identifier("MyStruct")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.Identifier("TypeA")),
            SimpleToken(lex.Identifier("A")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.Identifier("TypeB")),
            SimpleToken(lex.Identifier("B")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.RightBracket()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    with pytest.raises(parse.UnexpectedToken):
        parser.parse_struct()


def test_parse_struct_no_trailing_comma(simple_field_foo: SimpleFieldType) -> None:
    struct_ident = node.Identifier(Span(), "MyStruct")

    # struct MyStruct {
    #   FooType foo
    # };

    expected_node = node.Struct(
        Span(),
        struct_ident,
        [
            simple_field_foo.node,
        ],
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Struct)),
            SimpleToken(lex.Identifier("MyStruct")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.Identifier("FooType")),
            SimpleToken(lex.Identifier("foo")),
            SimpleToken(lex.RightBrace()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_struct() == expected_node


def test_parse_struct_two_field_no_trailing_comma(
    simple_field_foo: SimpleFieldType,
    simple_field_bar: SimpleFieldType,
) -> None:
    struct_ident = node.Identifier(Span(), "MyStruct")

    # struct MyStruct {
    #   FooType foo,
    #   BarType bar
    # };

    expected_node = node.Struct(
        Span(),
        struct_ident,
        [
            simple_field_foo.node,
            simple_field_bar.node,
        ],
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Struct)),
            SimpleToken(lex.Identifier("MyStruct")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.Identifier("FooType")),
            SimpleToken(lex.Identifier("foo")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.Identifier("BarType")),
            SimpleToken(lex.Identifier("bar")),
            SimpleToken(lex.RightBrace()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_struct() == expected_node


def test_parse_simple_enum() -> None:
    # enum Color {
    #   RED,
    # };

    expected_node = node.Enum(
        Span(), node.Identifier(Span(), "Color"), [node.Identifier(Span(), "RED")]
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Enum)),
            SimpleToken(lex.Identifier("Color")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.Identifier("RED")),
            SimpleToken(lex.Comma()),
            SimpleToken(lex.RightBrace()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_enum() == expected_node


def test_parse_empty_enum() -> None:
    # enum Color {};

    expected_node = node.Enum(Span(), node.Identifier(Span(), "Color"), [])

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Enum)),
            SimpleToken(lex.Identifier("Color")),
            SimpleToken(lex.LeftBrace()),
            SimpleToken(lex.RightBrace()),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_enum() == expected_node


def test_parse_multi_variant_enum(multi_variant_enum: MultiVariantEnumType) -> None:
    # enum Color {
    #   RED,
    #   GREEN,
    #   BLUE,
    # };

    parser = Parser(multi_variant_enum.tokens)

    assert parser.parse_enum() == multi_variant_enum.node


def test_parse_typedef_with_primitive_type(
    primitive_typedef: PrimitiveTypedefType,
) -> None:
    parser = Parser(primitive_typedef.tokens)

    assert parser.parse_typedef() == primitive_typedef.node


def test_parse_typedef_with_ident_type() -> None:
    # typedef FooType BarType

    expected_node = node.Typedef(
        Span(),
        node.Identifier(Span(), "BarType"),
        node.IdentifierType(Span(), node.Identifier(Span(), "FooType")),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.Typedef)),
            SimpleToken(lex.Identifier("FooType")),
            SimpleToken(lex.Identifier("BarType")),
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_typedef() == expected_node


def test_parse_definition(multi_field_struct: MultiFieldStructType) -> None:
    parser = Parser(multi_field_struct.tokens)

    assert parser.parse_definition() == multi_field_struct.node


def test_parse_multiple_definitions(
    primitive_typedef: PrimitiveTypedefType,
    multi_field_struct: MultiFieldStructType,
    multi_variant_enum: MultiVariantEnumType,
) -> None:
    tokens = (
        primitive_typedef.tokens + multi_field_struct.tokens + multi_variant_enum.tokens
    )
    expected_definitions = [
        primitive_typedef.node,
        multi_field_struct.node,
        multi_variant_enum.node,
    ]

    definitions: list[node.Definition] = []

    parser = Parser(tokens)

    definitions.append(parser.parse_definition())
    definitions.append(parser.parse_definition())
    definitions.append(parser.parse_definition())

    assert definitions == expected_definitions


def test_parse_module(
    primitive_typedef: PrimitiveTypedefType,
    multi_field_struct: MultiFieldStructType,
    multi_variant_enum: MultiVariantEnumType,
) -> None:
    tokens: list[lex.Token[lex.TokenKind]] = []

    tokens.append(SimpleToken(lex.Keyword(KeywordKind.Module)))
    tokens.append(SimpleToken(lex.Identifier("MyModule")))
    tokens.append(SimpleToken(lex.LeftBrace()))
    tokens += (
        primitive_typedef.tokens + multi_field_struct.tokens + multi_variant_enum.tokens
    )
    tokens.append(SimpleToken(lex.RightBrace()))
    tokens.append(SimpleToken(lex.SemiColon()))

    expected_node = node.Module(
        Span(),
        node.Identifier(Span(), "MyModule"),
        [
            primitive_typedef.node,
            multi_field_struct.node,
            multi_variant_enum.node,
        ],
    )

    parser = Parser(tokens)

    assert parser.parse_module() == expected_node
