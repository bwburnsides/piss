"""
Tests for functions in parse.py
"""

import pytest
import piss.parse as parse
from piss.parse import Parser, TypeVariant
import piss.lex as lex
from piss.lex import KeywordKind, Span, TokenKindVariant, TokenKindTag
from functools import partial
import typing
from typing import Callable

SimpleToken = partial(lex.Token, span=Span())


class SimpleFieldType(typing.NamedTuple):
    type: TokenKindVariant.Identifier
    name: TokenKindVariant.Identifier
    node: parse.Field


class PrimitiveTypedefType(typing.NamedTuple):
    tokens: list[lex.Token]
    node: parse.Typedef


class MultiVariantEnumType(typing.NamedTuple):
    tokens: list[lex.Token]
    node: parse.Enum


class MultiFieldStructType(typing.NamedTuple):
    tokens: list[lex.Token]
    node: parse.Struct


SimpleFieldFixtureType: Callable[
    [Callable[[], SimpleFieldType]],
    Callable[[], SimpleFieldType],
] = pytest.fixture


@SimpleFieldFixtureType
def simple_field() -> SimpleFieldType:
    type = SimpleToken(TokenKindVariant.Identifier("FooType"))
    if type.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    ident = SimpleToken(TokenKindVariant.Identifier("foo"))
    if ident.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    expected_node = parse.Field(
        span=Span(),
        kind=TypeVariant.Identifier(
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
        TypeVariant.Primitive(Span(), parse.PrimitiveKind.INT),
        parse.Identifier(Span(), "FooType"),
    )

    tokens = [
        SimpleToken(TokenKindVariant.Keyword(KeywordKind.TYPEDEF)),
        SimpleToken(TokenKindVariant.Keyword(KeywordKind.INT)),
        SimpleToken(TokenKindVariant.Identifier("FooType")),
        SimpleToken(TokenKindVariant.SemiColon()),
    ]

    return PrimitiveTypedefType(tokens, node)


MultiVariantEnumFxtureType: Callable[
    [Callable[[], MultiVariantEnumType]],
    Callable[[], MultiVariantEnumType],
] = pytest.fixture


@MultiVariantEnumFxtureType
def multi_variant_enum() -> MultiVariantEnumType:
    tokens = [
        SimpleToken(TokenKindVariant.Keyword(KeywordKind.ENUM)),
        SimpleToken(TokenKindVariant.Identifier("Color")),
        SimpleToken(TokenKindVariant.LeftBrace()),
        SimpleToken(TokenKindVariant.Identifier("RED")),
        SimpleToken(TokenKindVariant.Comma()),
        SimpleToken(TokenKindVariant.Identifier("GREEN")),
        SimpleToken(TokenKindVariant.Comma()),
        SimpleToken(TokenKindVariant.Identifier("BLUE")),
        SimpleToken(TokenKindVariant.Comma()),
        SimpleToken(TokenKindVariant.RightBrace()),
        SimpleToken(TokenKindVariant.SemiColon()),
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
        SimpleToken(TokenKindVariant.Keyword(KeywordKind.STRUCT)),
        SimpleToken(TokenKindVariant.Identifier("MyStruct")),
        SimpleToken(TokenKindVariant.LeftBrace()),
        SimpleToken(TokenKindVariant.Identifier("TypeA")),
        SimpleToken(TokenKindVariant.Identifier("A")),
        SimpleToken(TokenKindVariant.Comma()),
        SimpleToken(TokenKindVariant.Identifier("TypeB")),
        SimpleToken(TokenKindVariant.Identifier("B")),
        SimpleToken(TokenKindVariant.Comma()),
        SimpleToken(TokenKindVariant.RightBrace()),
        SimpleToken(TokenKindVariant.SemiColon()),
    ]

    type_a = SimpleToken(TokenKindVariant.Identifier("TypeA"))
    if type_a.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    ident_a = SimpleToken(TokenKindVariant.Identifier("A"))
    if ident_a.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    field_a = parse.Field(
        span=Span(),
        kind=TypeVariant.Identifier(
            span=type_a.span,
            type=parse.Identifier(type_a.span, type_a.kind.name),
        ),
        ident=parse.Identifier(ident_a.span, ident_a.kind.name),
    )

    type_b = SimpleToken(TokenKindVariant.Identifier("TypeB"))
    if type_b.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    ident_b = SimpleToken(TokenKindVariant.Identifier("B"))
    if ident_b.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    field_b = parse.Field(
        span=Span(),
        kind=TypeVariant.Identifier(
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
    expected_token = SimpleToken(TokenKindVariant.LeftBrace())
    parser = Parser([expected_token])

    token = parser.parse_token(TokenKindVariant.LeftBrace)
    assert token == expected_token


def test_parse_second_token() -> None:
    expected_token = SimpleToken(TokenKindVariant.RightBrace())

    parser = Parser([SimpleToken(TokenKindVariant.LeftBrace()), expected_token])
    parser.parse_token(TokenKindVariant.LeftBrace)

    assert parser.parse_token(TokenKindVariant.RightBrace) == expected_token


def test_parse_keyword() -> None:
    token = SimpleToken(TokenKindVariant.Keyword(KeywordKind.MODULE))
    expected_node = parse.Keyword(token.span, KeywordKind.MODULE)

    parser = Parser([token])

    assert parser.parse_keyword(KeywordKind.MODULE) == expected_node


def test_parse_second_keyword() -> None:
    token = SimpleToken(TokenKindVariant.Keyword(KeywordKind.MODULE))
    expected_node = parse.Keyword(token.span, KeywordKind.MODULE)

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.RightBrace()),
            token,
        ]
    )
    parser.parse_token(TokenKindVariant.RightBrace)

    assert parser.parse_keyword(KeywordKind.MODULE) == expected_node


def test_parse_identifier() -> None:
    token = SimpleToken(TokenKindVariant.Identifier("FooIdentifier"))
    if token.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    expected_node = parse.Identifier(token.span, token.kind.name)

    parser = Parser([token])
    assert parser.parse_identifier() == expected_node


def test_parse_second_identifier() -> None:
    token = SimpleToken(TokenKindVariant.Identifier("FooIdentifier"))
    if token.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    expected_node = parse.Identifier(token.span, token.kind.name)

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.RightBrace()),
            token,
        ]
    )
    parser.parse_token(TokenKindVariant.RightBrace)

    assert parser.parse_identifier() == expected_node


def test_parse_integer() -> None:
    token = SimpleToken(TokenKindVariant.Integer(5))
    if token.kind.tag is not TokenKindTag.INTEGER:
        raise ValueError

    expected_node = parse.Integer(token.span, token.kind.value)

    parser = Parser([token])

    assert parser.parse_integer() == expected_node


def test_parse_second_integer() -> None:
    token = SimpleToken(TokenKindVariant.Integer(5))
    if token.kind.tag is not TokenKindTag.INTEGER:
        raise ValueError

    expected_node = parse.Integer(token.span, token.kind.value)

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.LeftBrace()),
            token,
        ]
    )
    parser.parse_token(TokenKindVariant.LeftBrace)

    assert parser.parse_integer() == expected_node


def test_parse_integer_expression() -> None:
    token = SimpleToken(TokenKindVariant.Integer(5))
    if token.kind.tag is not TokenKindTag.INTEGER:
        raise ValueError

    expected_node = parse.Expression(
        token.span,
        parse.Integer(token.span, token.kind.value),
    )

    parser = Parser([token])

    assert parser.parse_expression() == expected_node


def test_parse_identifier_expression() -> None:
    token = SimpleToken(TokenKindVariant.Identifier("foo"))
    if token.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    expected_node = parse.Expression(
        token.span,
        parse.Identifier(token.span, token.kind.name),
    )

    parser = Parser([token])

    assert parser.parse_expression() == expected_node


def test_parse_primitive_type() -> None:
    token = SimpleToken(TokenKindVariant.Keyword(KeywordKind.INT))
    if token.kind.tag is not TokenKindTag.KEYWORD:
        raise ValueError

    expected_node = TypeVariant.Primitive(
        token.span,
        parse.PrimitiveKind.INT,
    )

    parser = Parser([token])

    assert parser.parse_type() == expected_node


def test_parse_identifier_type() -> None:
    token = SimpleToken(TokenKindVariant.Identifier("FooType"))
    if token.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError

    expected_node = TypeVariant.Identifier(
        token.span,
        parse.Identifier(token.span, token.kind.name),
    )

    parser = Parser([token])

    assert parser.parse_type() == expected_node


def test_parse_array_type() -> None:
    # int[5]

    expected_node = TypeVariant.Array(
        Span(),
        TypeVariant.Primitive(Span(), parse.PrimitiveKind.INT),
        parse.Expression(Span(), parse.Integer(Span(), 5)),
    )

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.INT)),
            SimpleToken(TokenKindVariant.LeftBracket()),
            SimpleToken(TokenKindVariant.Integer(5)),
            SimpleToken(TokenKindVariant.RightBracket()),
        ]
    )

    assert parser.parse_type() == expected_node


def test_parse_nested_array_type() -> None:
    # int[5][ConstFoo]

    expected_node = TypeVariant.Array(
        Span(),
        TypeVariant.Array(
            Span(),
            TypeVariant.Primitive(Span(), parse.PrimitiveKind.INT),
            parse.Expression(Span(), parse.Integer(Span(), 5)),
        ),
        parse.Expression(Span(), parse.Identifier(Span(), "ConstFoo")),
    )

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.INT)),
            SimpleToken(TokenKindVariant.LeftBracket()),
            SimpleToken(TokenKindVariant.Integer(5)),
            SimpleToken(TokenKindVariant.RightBracket()),
            SimpleToken(TokenKindVariant.LeftBracket()),
            SimpleToken(TokenKindVariant.Identifier("ConstFoo")),
            SimpleToken(TokenKindVariant.RightBracket()),
        ]
    )

    assert parser.parse_type() == expected_node


def test_parse_field_with_primitive_type() -> None:
    type = SimpleToken(TokenKindVariant.Keyword(KeywordKind.INT))
    ident = SimpleToken(TokenKindVariant.Identifier("foo"))
    if ident.kind.tag is not TokenKindTag.IDENTIFIER:
        raise ValueError
    expected_node = parse.Field(
        span=Span(),
        kind=TypeVariant.Primitive(type.span, parse.PrimitiveKind.INT),
        ident=parse.Identifier(ident.span, ident.kind.name),
    )

    parser = Parser([type, ident])

    assert parser.parse_field() == expected_node


def test_parse_field_with_identifier_type(simple_field: SimpleFieldType) -> None:
    type, ident, expected_node = simple_field

    parser = Parser([SimpleToken(type), SimpleToken(ident)])

    assert parser.parse_field() == expected_node


def test_parse_const(simple_field: SimpleFieldType) -> None:
    expr = SimpleToken(TokenKindVariant.Integer(5))
    if expr.kind.tag is not TokenKindTag.INTEGER:
        raise ValueError

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
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.CONST)),
            SimpleToken(simple_field.type),
            SimpleToken(simple_field.name),
            SimpleToken(TokenKindVariant.Equals()),
            expr,
            SimpleToken(TokenKindVariant.SemiColon()),
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
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.STRUCT)),
            SimpleToken(TokenKindVariant.Identifier("MyStruct")),
            SimpleToken(TokenKindVariant.LeftBrace()),
            SimpleToken(TokenKindVariant.Identifier("FooType")),
            SimpleToken(TokenKindVariant.Identifier("foo")),
            SimpleToken(TokenKindVariant.Comma()),
            SimpleToken(TokenKindVariant.RightBrace()),
            SimpleToken(TokenKindVariant.SemiColon()),
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
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.STRUCT)),
            SimpleToken(TokenKindVariant.Identifier("MyStruct")),
            SimpleToken(TokenKindVariant.LeftBrace()),
            SimpleToken(TokenKindVariant.RightBrace()),
            SimpleToken(TokenKindVariant.SemiColon()),
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
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.STRUCT)),
            SimpleToken(TokenKindVariant.Identifier("MyStruct")),
            SimpleToken(TokenKindVariant.LeftBrace()),
            SimpleToken(TokenKindVariant.Identifier("TypeA")),
            SimpleToken(TokenKindVariant.Identifier("A")),
            SimpleToken(TokenKindVariant.Comma()),
            SimpleToken(TokenKindVariant.Identifier("TypeB")),
            SimpleToken(TokenKindVariant.Identifier("B")),
            SimpleToken(TokenKindVariant.Comma()),
            SimpleToken(TokenKindVariant.RightBrace()),
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
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.STRUCT)),
            SimpleToken(TokenKindVariant.Identifier("MyStruct")),
            SimpleToken(TokenKindVariant.LeftBrace()),
            SimpleToken(TokenKindVariant.Identifier("TypeA")),
            SimpleToken(TokenKindVariant.Identifier("A")),
            SimpleToken(TokenKindVariant.Comma()),
            SimpleToken(TokenKindVariant.Identifier("TypeB")),
            SimpleToken(TokenKindVariant.Identifier("B")),
            SimpleToken(TokenKindVariant.Comma()),
            SimpleToken(TokenKindVariant.RightBracket()),
            SimpleToken(TokenKindVariant.SemiColon()),
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
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.ENUM)),
            SimpleToken(TokenKindVariant.Identifier("Color")),
            SimpleToken(TokenKindVariant.LeftBrace()),
            SimpleToken(TokenKindVariant.Identifier("RED")),
            SimpleToken(TokenKindVariant.Comma()),
            SimpleToken(TokenKindVariant.RightBrace()),
            SimpleToken(TokenKindVariant.SemiColon()),
        ]
    )

    assert parser.parse_enum() == expected_node


def test_parse_empty_enum() -> None:
    # enum Color {};

    expected_node = parse.Enum(Span(), parse.Identifier(Span(), "Color"), [])

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.ENUM)),
            SimpleToken(TokenKindVariant.Identifier("Color")),
            SimpleToken(TokenKindVariant.LeftBrace()),
            SimpleToken(TokenKindVariant.RightBrace()),
            SimpleToken(TokenKindVariant.SemiColon()),
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
        TypeVariant.Identifier(Span(), parse.Identifier(Span(), "FooType")),
        parse.Identifier(Span(), "BarType"),
    )

    parser = Parser(
        [
            SimpleToken(TokenKindVariant.Keyword(KeywordKind.TYPEDEF)),
            SimpleToken(TokenKindVariant.Identifier("FooType")),
            SimpleToken(TokenKindVariant.Identifier("BarType")),
            SimpleToken(TokenKindVariant.SemiColon()),
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

    definitions = []

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
    tokens = []

    tokens.append(SimpleToken(TokenKindVariant.Keyword(KeywordKind.MODULE)))
    tokens.append(SimpleToken(TokenKindVariant.Identifier("MyModule")))
    tokens.append(SimpleToken(TokenKindVariant.LeftBrace()))
    tokens += (
        primitive_typedef.tokens + multi_field_struct.tokens + multi_variant_enum.tokens
    )
    tokens.append(SimpleToken(TokenKindVariant.RightBrace()))
    tokens.append(SimpleToken(TokenKindVariant.SemiColon()))

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
