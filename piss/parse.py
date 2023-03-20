"""
Tools for parsing the PISS grammer.
"""

from dataclasses import dataclass
import enum
from piss import lex
import typing
from typing import Literal
from piss.lex import TokenKindVariant, TokenKindTag


@dataclass
class Node:
    """
    Represents a generic node in a PISS AST
    """

    span: lex.Span


@dataclass
class Keyword(Node):
    kind: lex.KeywordKind


@dataclass
class Integer(Node):
    value: int


@dataclass
class Identifier(Node):
    name: str


@dataclass
class Expression(Node):
    expr: Identifier | Integer


@dataclass
class PrimitiveKind(enum.Enum):
    """
    PrimitiveKind enumerates the primitive (builtin) types in PISS grammar.
    These are represented by Keyword tokens.
    """

    UINT = enum.auto()
    INT = enum.auto()


@enum.unique
class TypeTag(enum.Enum):
    """
    TypeTag enumerates the groups of types found in PISS grammar.
    """

    PRIMITIVE = enum.auto()
    IDENTIFIER = enum.auto()
    ARRAY = enum.auto()


class TypeVariant:
    @dataclass
    class Primitive(Node):
        type: PrimitiveKind
        tag: typing.Literal[TypeTag.PRIMITIVE] = TypeTag.PRIMITIVE

    @dataclass
    class Identifier(Node):
        type: Identifier
        tag: typing.Literal[TypeTag.IDENTIFIER] = TypeTag.IDENTIFIER

    @dataclass
    class Array(Node):
        type: "Type"
        length: Expression
        tag: typing.Literal[TypeTag.ARRAY] = TypeTag.ARRAY


Type = TypeVariant.Primitive | TypeVariant.Identifier | TypeVariant.Array


@dataclass
class Field(Node):
    kind: Type
    ident: Identifier


@enum.unique
class DefinitionTag(enum.Enum):
    CONST = enum.auto()
    STRUCT = enum.auto()
    ENUM = enum.auto()
    TYPEDEF = enum.auto()
    MODULE = enum.auto()


class DefinitionVariant:
    @dataclass
    class Const(Node):
        kind: Identifier | Type
        ident: Identifier
        expr: Expression
        tag: Literal[DefinitionTag.CONST] = DefinitionTag.CONST

    @dataclass
    class Struct(Node):
        ident: Identifier
        fields: list[Field]
        tag: Literal[DefinitionTag.STRUCT] = DefinitionTag.STRUCT

    @dataclass
    class Enum(Node):
        ident: Identifier
        variants: list[Identifier]
        tag: Literal[DefinitionTag.ENUM] = DefinitionTag.ENUM

    @dataclass
    class Typedef(Node):
        kind: Type
        ident: Identifier
        tag: Literal[DefinitionTag.TYPEDEF] = DefinitionTag.TYPEDEF

    @dataclass
    class Module(Node):
        ident: Identifier
        defs: list["Definition"]
        tag: Literal[DefinitionTag.MODULE] = DefinitionTag.MODULE


Definition = (
    DefinitionVariant.Const
    | DefinitionVariant.Struct
    | DefinitionVariant.Enum
    | DefinitionVariant.Typedef
    | DefinitionVariant.Module
)


class ParseError(ValueError):
    ...


class UnexpectedEOF(ParseError):
    ...


class UnexpectedToken(ParseError):
    ...


class Parser:
    def __init__(self, tokens: list[lex.Token]):
        self.tokens = tokens
        self.current_index = 0

    def peek(self) -> lex.TokenKind | None:
        try:
            current_token = self.tokens[self.current_index]
        except IndexError:
            return None

        return current_token.kind

    def next(self) -> lex.Token | None:
        token = self.tokens[self.current_index]

        if token is not None:
            self.current_index += 1

        return token

    def next_then_unwrap(self) -> lex.Token:
        token_or_none = self.next()

        if token_or_none is None:
            raise UnexpectedEOF

        return token_or_none

    def parse_token(self, kind: typing.Type[lex.TokenKind]) -> lex.Token:
        peeked = self.peek()

        if peeked is None:
            raise UnexpectedEOF

        if not isinstance(peeked, kind):
            raise UnexpectedToken

        return self.next_then_unwrap()

    def parse_keyword(self, kind: lex.KeywordKind) -> Keyword:
        peeked = self.peek()

        if peeked is None:
            raise UnexpectedEOF

        if not isinstance(peeked, TokenKindVariant.Keyword):
            raise UnexpectedToken

        if peeked.keyword != kind:
            raise UnexpectedToken

        next = self.next_then_unwrap()
        if next.kind.tag is not TokenKindTag.KEYWORD:
            raise UnexpectedToken

        return Keyword(next.span, next.kind.keyword)

    def parse_identifier(self) -> Identifier:
        peeked = self.peek()

        if peeked is None:
            raise UnexpectedEOF

        if not isinstance(peeked, TokenKindVariant.Identifier):
            raise UnexpectedToken

        next = self.next_then_unwrap()
        if next.kind.tag is not TokenKindTag.IDENTIFIER:
            raise UnexpectedToken

        return Identifier(next.span, next.kind.name)

    def parse_integer(self) -> Integer:
        peeked = self.peek()

        if peeked is None:
            raise UnexpectedEOF

        if not isinstance(peeked, TokenKindVariant.Integer):
            raise UnexpectedToken

        next = self.next_then_unwrap()
        if next.kind.tag is not TokenKindTag.INTEGER:
            raise UnexpectedEOF

        return Integer(next.span, next.kind.value)

    def parse_expression(self) -> Expression:
        # Expression = Identifier | Integer

        # Attempt to parse an Identitifer. If there is an UnexpectedToken, then we could
        # not successfully parse one, and now we'll try to parse an Integer.
        try:
            ident = self.parse_identifier()
        except UnexpectedToken:
            pass
        else:
            return Expression(ident.span, ident)

        # Don't try to catch UnexpectedToken here. If its raised that
        # means that neither an Identifier or an Integer could be parsed.
        # This means that no Expression could be parsed since we've
        # already tried to parse an Identifier.
        integer = self.parse_integer()
        return Expression(integer.span, integer)

    def parse_type(self) -> Type:
        # Type = PrimitiveType
        #       | Identifier
        #       | Type LeftBracket Expression RightBracket

        primitives = [
            (lex.KeywordKind.UINT, PrimitiveKind.UINT),
            (lex.KeywordKind.INT, PrimitiveKind.INT),
        ]

        parsed_type: Type | None = None

        # Search for a Keyword token corresponding to the PrimitiveType
        # KeywordKind. If there is an UnexpectedToken, then we'll try to parse
        # the next PrimitiveType KeywordKind. If there isn't, then we've parsed
        # the base of a Type and can progress to checking for array types.
        for keyword, primitive in primitives:
            try:
                parsed_primitive_type_keyword = self.parse_keyword(keyword)
            except UnexpectedToken:
                continue
            else:
                parsed_type = TypeVariant.Primitive(
                    parsed_primitive_type_keyword.span, primitive
                )

        # If no PrimitiveType could be parsed, then the Type base must be an Identifier.
        # This Identifier corresponds to the name of a user-defined type, maybe via
        # a typedef, struct, or enum definition. Don't try to catch an exceptions
        # since this is the last possibility for a valid Type base.
        if parsed_type is None:
            ident = self.parse_identifier()
            parsed_type = TypeVariant.Identifier(ident.span, ident)

        start_span = parsed_type.span

        # Now we check if this type is an array type.
        while True:
            # Loop with the following patter: look for left bracket, then look for experssion, then look for right bracket.
            # If no left bracket, we're done (break and return)
            # If anything after left bracket is missing, then unexpectedtoken / unexepectedeof

            try:
                self.parse_token(TokenKindVariant.LeftBracket)
            except ParseError:
                break

            expr = self.parse_expression()
            left_bracket = self.parse_token(TokenKindVariant.RightBracket)

            end_span = left_bracket.span

            parsed_type = TypeVariant.Array(
                start_span + end_span,
                parsed_type,
                expr,
            )

        return parsed_type

    def parse_field(self) -> Field:
        # Field = Type Identifier

        type = self.parse_type()
        ident = self.parse_identifier()

        return Field(type.span + ident.span, type, ident)

    def parse_const(self) -> DefinitionVariant.Const:
        # ConstDefinition = Keyword::CONST Field Equals Expression SemiColon

        const = self.parse_keyword(lex.KeywordKind.CONST)

        field = self.parse_field()

        self.parse_token(TokenKindVariant.Equals)

        expr = self.parse_expression()

        semicolon = self.parse_token(TokenKindVariant.SemiColon)

        return DefinitionVariant.Const(
            const.span + semicolon.span, field.kind, field.ident, expr
        )

    def parse_struct(self) -> DefinitionVariant.Struct:
        # StructDefinition = Keyword::STRUCT Identifier LeftBrace (Field Comma)* RightBrace SemiColon

        struct = self.parse_keyword(lex.KeywordKind.STRUCT)

        ident = self.parse_identifier()

        self.parse_token(TokenKindVariant.LeftBrace)

        # An enum can contain any number of fields in it, which means we need to loop in order to parse
        # them all. A variant is defined as an Identifier that must be followed by a Comma.
        fields = []
        while True:
            # Attempt to parse a Field. If there is an UnexpectedToken, then there is not a field at the
            # front of the Token stream and we are done parsing fields and can break.
            try:
                field = self.parse_field()
            except UnexpectedToken:
                break
            else:
                fields.append(field)

            # If we successfully parsed an Identifier then it must be followed by a Comma, do don't catch any
            # exceptions which may occur.
            self.parse_token(TokenKindVariant.Comma)

        self.parse_token(TokenKindVariant.RightBrace)
        semicolon = self.parse_token(TokenKindVariant.SemiColon)

        return DefinitionVariant.Struct(struct.span + semicolon.span, ident, fields)

    def parse_enum(self) -> DefinitionVariant.Enum:
        # EnumDefinition = Keyword::ENUM Identifier LeftBrace (Identifier Comma)* RightBrace SemiColon

        enum = self.parse_keyword(lex.KeywordKind.ENUM)

        ident = self.parse_identifier()

        self.parse_token(TokenKindVariant.LeftBrace)

        # An enum can contain any number of variants in it, which means we need to loop in order to parse
        # them all. A variant is defined as an Identifier that must be followed by a Comma.
        variants = []
        while True:
            # Attempt to parse an Identifier, which names the variant. If there is an UnexpectedToken, then
            # there is not a variant at the front of the Token stream and we are done parsing variants and can break.
            try:
                variant = self.parse_identifier()
            except UnexpectedToken:
                break
            else:
                variants.append(variant)

            # If we successfully parsed an Identifier then it must be followed by a Comma, so don't catch any
            # exceptions which may occur.
            self.parse_token(TokenKindVariant.Comma)

        self.parse_token(TokenKindVariant.RightBrace)
        semicolon = self.parse_token(TokenKindVariant.SemiColon)

        return DefinitionVariant.Enum(enum.span + semicolon.span, ident, variants)

    def parse_typedef(self) -> DefinitionVariant.Typedef:
        # TypdefDefinition = Keyword::TYPEDEF Type Identifier SemiColon

        # TODO: use Field instead of Type Identifier and update grammar

        typedef = self.parse_keyword(lex.KeywordKind.TYPEDEF)

        field = self.parse_field()

        semicolon = self.parse_token(TokenKindVariant.SemiColon)

        return DefinitionVariant.Typedef(
            typedef.span + semicolon.span, field.kind, field.ident
        )

    def parse_definition(self) -> Definition:
        definition_parsers: list[typing.Callable[[], Definition]] = [
            self.parse_const,
            self.parse_struct,
            self.parse_enum,
            self.parse_typedef,
        ]

        # Try to parse each type of definition. If it succeeds, then return that definition.
        # However, a ParseError might be raised. These include UnexpectedEOF and UnexpectedToken.
        # In the case of an UnexecptedEOF, that means that the token stream was unexpectedly
        # exhausted. This is a syntactic error, it means that tokens were missing. Maybe a missing
        # semicolon, so we raise that exception to our caller. However, if for example an
        # UnexpectedToken exception occurs, this likely means that the token stream doesn't
        # currenly have the target definition at the front. In that case, we ignore the
        # exception and continue onwards and attempt to parse the next type of definition.

        for parser in definition_parsers:
            try:
                return parser()
            except UnexpectedEOF:
                raise UnexpectedEOF
            except UnexpectedToken:
                continue

        raise UnexpectedToken

    def parse_module(self) -> DefinitionVariant.Module:
        # ModuleDefinition = Keyword::MODULE Identifier LeftBrace (Definitions)* RightBrace SemiColon

        module = self.parse_keyword(lex.KeywordKind.MODULE)

        ident = self.parse_identifier()

        self.parse_token(TokenKindVariant.LeftBrace)

        definitions = []

        while True:
            try:
                definition = self.parse_definition()
            except UnexpectedToken:
                break
            except ParseError:
                raise ParseError
            else:
                definitions.append(definition)

        self.parse_token(TokenKindVariant.RightBrace)
        semicolon = self.parse_token(TokenKindVariant.SemiColon)

        return DefinitionVariant.Module(
            module.span + semicolon.span, ident, definitions
        )
