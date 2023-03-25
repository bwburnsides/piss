"""
Tests for functions in parse.py
"""

import pytest
import piss.parse as parse
from piss.parse import Parser
import piss.lex as lex
from piss.lex import KeywordKind, Span
from typing import Callable, NamedTuple
from functools import partial

SimpleToken = partial(lex.Token, span=Span())


class SimpleFieldType(NamedTuple):
    type: lex.Identifier
    name: lex.Identifier
    node: parse.Field


class PrimitiveTypedefType(NamedTuple):
    tokens: list[lex.Token]
    node: parse.Typedef


class MultiVariantEnumType(NamedTuple):
    tokens: list[lex.Token]
    node: parse.Enum


class MultiFieldStructType(NamedTuple):
    tokens: list[lex.Token]
    node: parse.Struct


SimpleFieldFixtureType: Callable[
    [Callable[[], SimpleFieldType]],
    Callable[[], SimpleFieldType],
] = pytest.fixture


@SimpleFieldFixtureType
def simple_field() -> SimpleFieldType:
    type = SimpleToken(lex.Identifier("FooType"))
    assert isinstance(type.kind, lex.Identifier)

    ident = SimpleToken(lex.Identifier("foo"))
    assert isinstance(ident.kind, lex.Identifier)

    expected_node = parse.Field(
        span=Span(),
        kind=parse.IdentifierType(
            type.span,
            parse.Identifier(type.span, type.kind.name),
        ),
        ident=parse.Identifier(ident.span, ident.kind.name),
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
    node = parse.Typedef(
        Span(),
        parse.PrimitiveType(Span(), parse.PrimitiveKind.INT),
        parse.Identifier(Span(), "FooType"),
    )

    tokens = [
        SimpleToken(lex.Keyword(KeywordKind.TYPEDEF)),
        SimpleToken(lex.Keyword(KeywordKind.INT)),
        SimpleToken(lex.Identifier("FooType")),
        SimpleToken(lex.SemiColon()),
    ]

    return PrimitiveTypedefType(tokens, node)


MultiVariantEnumFixtureType: Callable[
    [Callable[[], MultiVariantEnumType]],
    Callable[[], MultiVariantEnumType],
] = pytest.fixture


@MultiVariantEnumFixtureType
def multi_variant_enum() -> MultiVariantEnumType:
    tokens = [
        SimpleToken(lex.Keyword(KeywordKind.ENUM)),
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

    node = parse.Enum(
        Span(),
        parse.Identifier(Span(), "Color"),
        [
            parse.Identifier(Span(), "RED"),
            parse.Identifier(Span(), "GREEN"),
            parse.Identifier(Span(), "BLUE"),
        ],
    )

    return MultiVariantEnumType(tokens, node)


MultiFieldStructFixtureType: Callable[
    [Callable[[], MultiFieldStructType]],
    Callable[[], MultiFieldStructType],
] = pytest.fixture


@MultiFieldStructFixtureType
def multi_field_struct() -> MultiFieldStructType:
    tokens = [
        SimpleToken(lex.Keyword(KeywordKind.STRUCT)),
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

    field_a = parse.Field(
        span=Span(),
        kind=parse.IdentifierType(
            span=type_a.span,
            type=parse.Identifier(type_a.span, type_a.kind.name),
        ),
        ident=parse.Identifier(ident_a.span, ident_a.kind.name),
    )

    type_b = SimpleToken(lex.Identifier("TypeB"))
    assert isinstance(type_b.kind, lex.Identifier)

    ident_b = SimpleToken(lex.Identifier("B"))
    assert isinstance(ident_b.kind, lex.Identifier)

    field_b = parse.Field(
        span=Span(),
        kind=parse.IdentifierType(
            span=type_b.span,
            type=parse.Identifier(type_b.span, type_b.kind.name),
        ),
        ident=parse.Identifier(ident_b.span, ident_b.kind.name),
    )

    node = parse.Struct(
        Span(),
        parse.Identifier(Span(), "MyStruct"),
        [field_a, field_b],
    )

    return MultiFieldStructType(tokens, node)


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
    token = SimpleToken(lex.Keyword(KeywordKind.MODULE))
    expected_node = parse.Keyword(token.span, KeywordKind.MODULE)

    parser = Parser([token])

    assert parser.parse_keyword(KeywordKind.MODULE) == expected_node


def test_parse_second_keyword() -> None:
    token = SimpleToken(lex.Keyword(KeywordKind.MODULE))
    expected_node = parse.Keyword(token.span, KeywordKind.MODULE)

    parser = Parser(
        [
            SimpleToken(lex.RightBrace()),
            token,
        ]
    )
    parser.parse_token(lex.RightBrace)

    assert parser.parse_keyword(KeywordKind.MODULE) == expected_node


def test_parse_identifier() -> None:
    token = SimpleToken(lex.Identifier("FooIdentifier"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = parse.Identifier(token.span, token.kind.name)

    parser = Parser([token])
    assert parser.parse_identifier() == expected_node


def test_parse_second_identifier() -> None:
    token = SimpleToken(lex.Identifier("FooIdentifier"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = parse.Identifier(token.span, token.kind.name)

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

    expected_node = parse.Integer(token.span, token.kind.value)

    parser = Parser([token])

    assert parser.parse_integer() == expected_node


def test_parse_second_integer() -> None:
    token = SimpleToken(lex.Integer(5))
    assert isinstance(token.kind, lex.Integer)

    expected_node = parse.Integer(token.span, token.kind.value)

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

    expected_node = parse.Expression(
        token.span,
        parse.Integer(token.span, token.kind.value),
    )

    parser = Parser([token])

    assert parser.parse_expression() == expected_node


def test_parse_identifier_expression() -> None:
    token = SimpleToken(lex.Identifier("foo"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = parse.Expression(
        token.span,
        parse.Identifier(token.span, token.kind.name),
    )

    parser = Parser([token])

    assert parser.parse_expression() == expected_node


def test_parse_primitive_type() -> None:
    token = SimpleToken(lex.Keyword(KeywordKind.INT))

    expected_node = parse.PrimitiveType(
        token.span,
        parse.PrimitiveKind.INT,
    )

    parser = Parser([token])

    assert parser.parse_type() == expected_node


def test_parse_identifier_type() -> None:
    token = SimpleToken(lex.Identifier("FooType"))
    assert isinstance(token.kind, lex.Identifier)

    expected_node = parse.IdentifierType(
        token.span,
        parse.Identifier(token.span, token.kind.name),
    )

    parser = Parser([token])

    assert parser.parse_type() == expected_node


def test_parse_array_type() -> None:
    # int[5]

    expected_node = parse.ArrayType(
        Span(),
        parse.PrimitiveType(Span(), parse.PrimitiveKind.INT),
        parse.Expression(Span(), parse.Integer(Span(), 5)),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.INT)),
            SimpleToken(lex.LeftBracket()),
            SimpleToken(lex.Integer(5)),
            SimpleToken(lex.RightBracket()),
        ]
    )

    assert parser.parse_type() == expected_node


def test_parse_nested_array_type() -> None:
    # int[5][ConstFoo]

    expected_node = parse.ArrayType(
        Span(),
        parse.ArrayType(
            Span(),
            parse.PrimitiveType(Span(), parse.PrimitiveKind.INT),
            parse.Expression(Span(), parse.Integer(Span(), 5)),
        ),
        parse.Expression(Span(), parse.Identifier(Span(), "ConstFoo")),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.INT)),
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
    type = SimpleToken(lex.Keyword(KeywordKind.INT))
    ident = SimpleToken(lex.Identifier("foo"))
    assert isinstance(ident.kind, lex.Identifier)

    expected_node = parse.Field(
        span=Span(),
        kind=parse.PrimitiveType(type.span, parse.PrimitiveKind.INT),
        ident=parse.Identifier(ident.span, ident.kind.name),
    )

    parser = Parser([type, ident])

    assert parser.parse_field() == expected_node


def test_parse_field_with_identifier_type(simple_field: SimpleFieldType) -> None:
    type, ident, expected_node = simple_field

    parser = Parser([SimpleToken(type), SimpleToken(ident)])

    assert parser.parse_field() == expected_node


def test_parse_const(simple_field: SimpleFieldType) -> None:
    expr = SimpleToken(lex.Integer(5))
    assert isinstance(expr.kind, lex.Integer)

    # FooType foo = 5;

    expected_node = parse.Const(
        Span(),
        simple_field.node.kind,
        simple_field.node.ident,
        parse.Expression(
            Span(),
            parse.Integer(Span(), expr.kind.value),
        ),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.CONST)),
            SimpleToken(simple_field.type),
            SimpleToken(simple_field.name),
            SimpleToken(lex.Equals()),
            expr,
            SimpleToken(lex.SemiColon()),
        ]
    )

    assert parser.parse_const() == expected_node


def test_parse_simple_struct(simple_field: SimpleFieldType) -> None:
    struct_ident = parse.Identifier(Span(), "MyStruct")

    # struct MyStruct {
    #   FooType foo,
    # };

    expected_node = parse.Struct(
        Span(),
        struct_ident,
        [
            simple_field.node,
        ],
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.STRUCT)),
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

    expected_node = parse.Struct(
        Span(),
        parse.Identifier(Span(), "MyStruct"),
        [],
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.STRUCT)),
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
            SimpleToken(lex.Keyword(KeywordKind.STRUCT)),
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
            SimpleToken(lex.Keyword(KeywordKind.STRUCT)),
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


def test_parse_simple_enum() -> None:
    # enum Color {
    #   RED,
    # };

    expected_node = parse.Enum(
        Span(), parse.Identifier(Span(), "Color"), [parse.Identifier(Span(), "RED")]
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.ENUM)),
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

    expected_node = parse.Enum(Span(), parse.Identifier(Span(), "Color"), [])

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.ENUM)),
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

    expected_node = parse.Typedef(
        Span(),
        parse.IdentifierType(Span(), parse.Identifier(Span(), "FooType")),
        parse.Identifier(Span(), "BarType"),
    )

    parser = Parser(
        [
            SimpleToken(lex.Keyword(KeywordKind.TYPEDEF)),
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

    definitions: list[parse.Definition] = []

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
    tokens: list[lex.Token] = []

    tokens.append(SimpleToken(lex.Keyword(KeywordKind.MODULE)))
    tokens.append(SimpleToken(lex.Identifier("MyModule")))
    tokens.append(SimpleToken(lex.LeftBrace()))
    tokens += (
        primitive_typedef.tokens + multi_field_struct.tokens + multi_variant_enum.tokens
    )
    tokens.append(SimpleToken(lex.RightBrace()))
    tokens.append(SimpleToken(lex.SemiColon()))

    expected_node = parse.Module(
        Span(),
        parse.Identifier(Span(), "MyModule"),
        [
            primitive_typedef.node,
            multi_field_struct.node,
            multi_variant_enum.node,
        ],
    )

    parser = Parser(tokens)

    assert parser.parse_module() == expected_node
