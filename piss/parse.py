# Expression = Identifier | Integer
#
# Type = Keyword(Int)
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
from enum import Enum, auto
import lex


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


class PrimitiveType(Enum):
    UINT = auto()
    INT = auto()


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
        try:
            return self.parse_const()
        except UnexpectedEOF:
            raise UnexpectedEOF
        except ParseError:
            pass

        try:
            return self.parse_struct()
        except UnexpectedEOF:
            raise UnexpectedEOF
        except ParseError:
            pass

        try:
            return self.parse_enum()
        except UnexpectedEOF:
            raise UnexpectedEOF
        except ParseError:
            pass

        try:
            return self.parse_typedef()
        except UnexpectedEOF:
            raise UnexpectedEOF
        except ParseError:
            pass

        try:
            return self.parse_module()
        except UnexpectedEOF:
            raise UnexpectedEOF
        except ParseError:
            pass

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

        fields = []

        while True:
            try:
                field = self.parse_field()
            except UnexpectedEOF:
                raise UnexpectedEOF
            except ParseError:
                break
            else:
                fields.append(field)

            try:
                self.parse_comma()
            except UnexpectedToken:
                break
            except ParseError:
                raise ParseError

        self.parse_token(lex.RightBrace)
        self.parse_token(lex.SemiColon)

        return Struct(0, ident, fields)

    def parse_enum(self) -> Enum:
        # EnumDefinition = Keyword::ENUM Identifier LeftBrace (Identifier Comma)* RightBrace SemiColon

        self.parse_keyword(lex.KeywordKind.ENUM)

        ident = self.parse_identifier()

        self.parse_token(lex.LeftBrace)

        variants = []

        while True:
            try:
                variant = self.parse_identifier()
            except UnexpectedEOF:
                raise UnexpectedEOF
            except ParseError:
                break
            else:
                variants.append(variant)

            try:
                self.parse_comma()
            except UnexpectedToken:
                break
            except ParseError:
                raise ParseError

        self.parse_token(lex.RightBrace)
        self.parse_token(lex.SemiColon)

        return Enum(0, ident, variants)

    def parse_typedef(self) -> Typedef:
        # TypdefDefinition = Keyword::TYPEDEF Type Identifier SemiColon

        self.parse(lex.KeywordKind.TYPEDEF)

        kind = self.parse_type()

        ident = self.parse_identifier()

        self.parse_token(lex.SemiColon)

        return Typedef(0, kind, ident)

    def parse_module(self) -> Typedef:
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

        primitives = [
            (lex.KeywordKind.UINT, PrimitiveType.UINT),
            (lex.KeywordKind.INT, PrimitiveType.INT),
        ]

        for token, primitive in primitives:
            try:
                self.parse_keyword(token)
            except ParseError:
                continue
            else:
                return Type(0, primitive)

        ident = self.parse_identifier

        # TODO: handle array types

        return Type(0, ident)

    def parse_expression(self) -> Expression:
        # Expression = Identifier | Integer
        try:
            ident = self.parse_identifier()
        except ParseError:
            pass
        else:
            return Expression(0, ident.kind.name)

    def parse_token(self, token: lex.TokenKind):
        ...

    def parse_keyword(self, keyword: lex.KeywordKind) -> lex.Keyword:
        ...

    def parse_identifier(self) -> Identifier:
        ...
