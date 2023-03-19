# Expression = Identifier | Integer
#
# PrimitiveType = Keyword::INT
#               | Keyword::UINT
#
# Type = PrimitiveType
#       | Identifier
#       | Type LeftBracket Expression RightBracket
#
# Field = Type Identifier
#
# ConstDefinition = Keyword::CONST Type Identifier Equals Expression SemiColon
#
# StructDefinition = Keyword::STRUCT Identifier LeftBrace (Field Comma)* RightBrace SemiColon
#
# EnumDefinition = Keyword::ENUM Identifier LeftBrace (Identifier Comma)* RightBrace SemiColon
#
# TypdefDefinition = Keyword::TYPEDEF Type Identifier SemiColon
#
# ModuleDefinition = Keyword::MODULE Identifier LeftBrace (Definitions)* RightBrace SemiColon

from dataclasses import dataclass
import enum
import piss.lex as lex
import typing


@dataclass
class Node:
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


class PrimitiveType(enum.Enum):
    UINT = enum.auto()
    INT = enum.auto()


@dataclass
class Type(Node):
    name: PrimitiveType | Identifier


Expression = Identifier | Integer


class Field:
    kind: Type
    ident: Identifier


@dataclass
class Definition(Node):
    pass


@dataclass
class Const(Definition):
    kind: Identifier | Type
    ident: Identifier
    expr: Expression


@dataclass
class Struct(Definition):
    ident: Identifier
    fields: list[Field]


@dataclass
class Enum(Definition):
    ident: Identifier
    variants: list[Identifier]


@dataclass
class Typedef(Definition):
    kind: Type
    ident: Identifier


@dataclass
class Module(Definition):
    ident: Identifier
    defs: list[Definition]


class ParseError(ValueError):
    pass


class UnexpectedEOF(ParseError):
    pass


class UnexpectedToken(ParseError):
    pass


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

    def parse_definition(self) -> Definition:
        definition_parsers = [
            self.parse_const,
            self.parse_struct,
            self.parse_enum,
            self.parse_typedef,
            self.parse_module,
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

    def parse_const(self) -> Const:
        # ConstDefinition = Keyword::CONST Type Identifier Equals Expression SemiColon

        self.parse_keyword(lex.KeywordKind.CONST)

        kind = self.parse_type()
        ident = self.parse_identifier()

        self.parse_token(lex.Equals)

        expr = self.parse_expression()

        self.parse_token(lex.SemiColon)

        # Where does the span come from again?
        return Const(0, kind, ident, expr)

    def parse_struct(self) -> Struct:
        # StructDefinition = Keyword::STRUCT Identifier LeftBrace (Field Comma)* RightBrace SemiColon

        self.parse_keyword(lex.KeywordKind.STRUCT)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

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
            self.parse_token(lex.Comma)

        self.parse_token(lex.RightBrace)
        self.parse_token(lex.SemiColon)

        return Struct(0, ident, fields)

    def parse_enum(self) -> Enum:
        # EnumDefinition = Keyword::ENUM Identifier LeftBrace (Identifier Comma)* RightBrace SemiColon

        self.parse_keyword(lex.KeywordKind.ENUM)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

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
            self.parse_token(lex.Comma)

        self.parse_token(lex.RightBrace)
        self.parse_token(lex.SemiColon)

        return Enum(0, ident, variants)

    def parse_typedef(self) -> Typedef:
        # TypdefDefinition = Keyword::TYPEDEF Type Identifier SemiColon

        self.parse_keyword(lex.KeywordKind.TYPEDEF)

        kind = self.parse_type()

        ident = self.parse_identifier()

        self.parse_token(lex.SemiColon)

        return Typedef(0, kind, ident)

    def parse_module(self) -> Module:
        # ModuleDefinition = Keyword::MODULE Identifier LeftBrace (Definitions)* RightBrace SemiColon

        self.parse_keyword(lex.KeywordKind.MODULE)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

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

        self.parse_token(lex.RightBrace)
        self.parse_token(lex.SemiColon)

        return Module(0, ident, definitions)

    def parse_type(self) -> Type:
        # Type = Keyword(Int)
        #       | Identifier
        #       | Type LeftBracket Expression RightBracket

        # TODO: HANDLE ARRAY TYPES

        primitives = [
            (lex.KeywordKind.UINT, PrimitiveType.UINT),
            (lex.KeywordKind.INT, PrimitiveType.INT),
        ]

        # Search for a Keyword token corresponding to the PrimitiveType
        # KeywordKind. If there is an UnexpectedToken, then we'll try to parse
        # the next PrimitiveType KeywordKind. If there isn't, then we've parsed
        # a Type and can return.
        for token, primitive in primitives:
            try:
                self.parse_keyword(token)
            except UnexpectedToken:
                continue
            else:
                return Type(0, primitive)

        # If no PrimitiveType could be parsed, then the Type must be an Identifier.
        # This Identifier corresponds to the name of a user-defined type, maybe via
        # a typedef, struct, or enum definition. Don't try to catch an exceptions
        # since this is the last possibility for a valid Type.
        ident = self.parse_identifier()

        return Type(0, ident)

    def parse_field(self) -> Field:
        raise NotImplementedError

    def parse_expression(self) -> Expression:
        # Expression = Identifier | Integer

        # Attempt to parse an Identitifer. If there is an UnexpectedToken, then we could
        # not successfully parse one, and now we'll try to parse an Integer.
        try:
            return self.parse_identifier()
        except UnexpectedToken:
            pass

        # Don't try to catch UnexpectedToken here. If its raised that
        # means that neither an Identifier or an Integer could be parsed.
        # This means that no Expression could be parsed since we've
        # already tried to parse an Identifier.
        return self.parse_integer()

    def parse_token(self, kind: typing.Type[lex.TokenKind]) -> lex.Token:
        raise NotImplementedError

    def parse_keyword(self, kind: lex.KeywordKind) -> lex.Keyword:
        peeked = self.peek()

        if peeked is None:
            raise UnexpectedEOF

        if not isinstance(peeked, Keyword):
            raise UnexpectedToken

        if peeked.kind != kind:
            raise UnexpectedToken

        return Keyword(0, kind)

    def parse_identifier(self) -> Identifier:
        raise NotImplementedError
