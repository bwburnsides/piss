from dataclasses import dataclass
from abc import ABC, abstractmethod
import enum
import typing

from piss import lex


T = typing.TypeVar("T")


@dataclass
class Node(ABC):
    """
    Represents a generic node in a PISS AST.
    """

    span: lex.Span

    @abstractmethod
    def accept(self, visitor: "NodeVisitor[T]") -> T:
        raise NotImplementedError


@dataclass
class Keyword(Node):
    kind: lex.KeywordKind

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_keyword(self)


@dataclass
class Integer(Node):
    value: int

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_integer(self)


@dataclass
class Identifier(Node):
    name: str

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_identifier(self)


@dataclass
class Expression(Node):
    expr: Identifier | Integer

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_expression(self)


class PrimitiveKind(enum.Enum):
    """
    PrimitiveKind enumerates the primitive (builtin) types in PISS grammar.
    These are represented by Keyword tokens.
    """

    Uint = enum.auto()
    Int = enum.auto()


@dataclass
class Type(Node):
    ...


@dataclass
class PrimitiveType(Type):
    type: PrimitiveKind

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_primitive_type(self)


@dataclass
class IdentifierType(Type):
    type: Identifier

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_identifier_type(self)


@dataclass
class ArrayType(Type):
    type: Type
    length: Expression

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_array_type(self)


@dataclass
class Field(Node):
    kind: Type
    ident: Identifier

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_field(self)


@dataclass
class Definition(Node):
    ident: Identifier


@dataclass
class Const(Definition):
    kind: Type
    expr: Expression

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_const(self)


@dataclass
class Struct(Definition):
    fields: list[Field]

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_struct(self)


@dataclass
class Enum(Definition):
    variants: list[Identifier]

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_enum(self)


@dataclass
class Typedef(Definition):
    kind: Type

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_typedef(self)


@dataclass
class Module(Definition):
    ident: Identifier
    definitions: list["Definition"]

    def accept(self, visitor: "NodeVisitor[T]") -> T:
        return visitor.visit_module(self)


class NodeVisitor(ABC, typing.Generic[T]):
    """
    Abstract Visitor base for parse.Node visitors. Implementors of this class can use
    it in order to visit all Nodes in a given tree. For example, a PrinterVisitor can
    be written in order to print all nodes in a given tree. Usage should look like this:

    visitor: NodeVisitor = ...
    node: Node = ...

    node.accept(visitor)
    """

    @abstractmethod
    def visit_keyword(self, keyword: Keyword) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_integer(self, integer: Integer) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_identifier(self, ident: Identifier) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_expression(self, expr: Expression) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_primitive_type(self, type: PrimitiveType) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_identifier_type(self, type: IdentifierType) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_array_type(self, type: ArrayType) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_field(self, field: Field) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_typedef(self, typedef: Typedef) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_const(self, const: Const) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_enum(self, enum: Enum) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_struct(self, struct: Struct) -> T:
        raise NotImplementedError

    @abstractmethod
    def visit_module(self, module: Module) -> T:
        raise NotImplementedError
