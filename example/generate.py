from piss import lex
from piss import parse
import typing

T = typing.TypeVar("T")

sample = """
module Foo {
    typedef int MyInt;

    struct Person {
        uint age,
        int debt
    };

    enum Color {
        Red, Green, Blue
    };

    const int THREE = 3;

    struct Span {
        int start,
        int stop,
    };

    enum TokenKind {
        LeftBrace, RightBrace, SemiColon
    };

    struct Token {
        Span span,
        TokenKind kind
    };

    typedef int[THREE] three_array;
    typedef int[THREE][10] matrix_type;

    struct SomeStruct {
        int Foo,
        int Bar,
        uint Baz
    };
};

// module Bar {};
"""


class Printer(parse.NodeVisitor):
    def __init__(self) -> None:
        self.indent_level = 0
        self.output = ""

    def print(self, value: str, indent: bool = False) -> None:
        lines = value.split("\n")

        if len(lines) > 0:
            self.output += lines[0]

        lines = lines[1:]

        if indent:
            self.indent()

        for line in lines:
            self.output += "\n" + self.indentation()
            self.output += line

    def indent(self) -> None:
        self.indent_level += 1
        self.newline()

    def dedent(self) -> None:
        self.indent_level -= 1
        self.newline()

    def indentation(self) -> str:
        return "    " * self.indent_level

    def newline(self) -> None:
        self.output += "\n" + self.indentation()

    def visit_keyword(self, keyword: parse.Keyword) -> None:
        ...

    def visit_integer(self, integer: parse.Integer) -> None:
        self.print(str(integer.value))

    def visit_identifier(self, ident: parse.Identifier) -> None:
        self.print(ident.name)

    def visit_expression(self, expr: parse.Expression) -> None:
        expr.expr.accept(self)

    def visit_primitive_type(self, type: parse.PrimitiveType) -> None:
        self.print(type.type.name)

    def visit_identifier_type(self, type: parse.IdentifierType) -> None:
        self.print(type.type.name)

    def visit_array_type(self, type: parse.ArrayType) -> None:
        self.print("list[")
        type.type.accept(self)
        self.print("]")

    def visit_field(self, field: parse.Field) -> None:
        field.ident.accept(self)
        self.print(": ")
        field.kind.accept(self)

    def visit_typedef(self, typedef: parse.Typedef) -> None:
        typedef.ident.accept(self)
        self.print(' = typing.NewType("')
        typedef.ident.accept(self)
        self.print('", ')
        typedef.kind.accept(self)
        self.print(")")
        self.newline()
        self.newline()

    def visit_const(self, const: parse.Const) -> None:
        const.ident.accept(self)
        self.print(": ")
        const.kind.accept(self)
        self.print(" = ")
        const.expr.accept(self)
        self.newline()
        self.newline()

    def visit_enum(self, enum: parse.Enum) -> None:
        self.print("class ")
        enum.ident.accept(self)
        self.print("(enum.Enum):", indent=True)

        for variant in enum.variants:
            variant.accept(self)
            self.print(" = enum.auto()")
            self.newline()

        self.dedent()

    def visit_struct(self, struct: parse.Struct) -> None:
        self.print("@dataclass\nclass ")
        struct.ident.accept(self)
        self.print(":", indent=True)

        for field in struct.fields:
            field.accept(self)
            self.newline()

        self.dedent()

    def visit_module(self, module: parse.Module) -> None:
        module.accept(self)


def main() -> None:
    tokens = lex.tokenize(sample)
    modules = parse.parse(tokens)
    print(len(modules))


if __name__ == "__main__":
    main()
