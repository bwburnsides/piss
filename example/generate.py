import typing
import textwrap
from functools import partial

from piss import lex, node, parse

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


class CodeGenerationError(ValueError):
    ...


class Printer(node.NodeVisitor[str]):
    indent = partial(textwrap.indent, prefix=" " * 4)

    def visit_keyword(self, keyword: node.Keyword) -> str:
        raise CodeGenerationError

    def visit_integer(self, integer: node.Integer) -> str:
        return str(integer.value)

    def visit_identifier(self, ident: node.Identifier) -> str:
        return ident.name

    def visit_expression(self, expr: node.Expression) -> str:
        return expr.expr.accept(self)

    def visit_primitive_type(self, type: node.PrimitiveType) -> str:
        repr_table = {
            node.PrimitiveKind.Uint: "int",
            node.PrimitiveKind.Int: "int",
        }

        try:
            return repr_table[type.type]
        except KeyError:
            raise CodeGenerationError

    def visit_identifier_type(self, type: node.IdentifierType) -> str:
        return type.type.accept(self)

    def visit_array_type(self, type: node.ArrayType) -> str:
        return f"list[{type.type.accept(self)}]"

    def visit_field(self, field: node.Field) -> str:
        template = "{name}: {type}"

        return template.format(
            name=field.ident.accept(self),
            type=field.kind.accept(self),
        )

    def visit_typedef(self, typedef: node.Typedef) -> str:
        type_template = '{name} = typing.NewType("{name}", {type})'

        return type_template.format(
            name=typedef.ident.accept(self),
            type=typedef.kind.accept(self),
        )

    def visit_const(self, const: node.Const) -> str:
        template = "{name}: {type} = {definition}"

        return template.format(
            name=const.ident.accept(self),
            type=const.kind.accept(self),
            definition=const.expr.accept(self),
        )

    def visit_enum(self, enum: node.Enum) -> str:
        declaration_template = "class {name}(enum.Enum):"

        repr = [declaration_template.format(name=enum.ident.accept(self))]
        repr.extend(self.indent(variant.accept(self)) for variant in enum.variants)

        return "\n".join(repr)

    def visit_struct(self, struct: node.Struct) -> str:
        declaration_template = "class {name}:"

        repr = [
            "@dataclass",
            declaration_template.format(name=struct.ident.accept(self)),
        ]
        repr.extend(self.indent(field.accept(self)) for field in struct.fields)

        return "\n".join(repr)

    def visit_module(self, module: node.Module) -> str:
        definition_reprs = [
            definition.accept(self) for definition in module.definitions
        ]

        return "\n\n".join(definition_reprs)


def main() -> None:
    tokens = lex.tokenize(sample)
    modules = parse.parse(tokens)
    printer = Printer()

    module_reprs = [module.accept(printer) for module in modules]
    for repr in module_reprs:
        print(repr)


if __name__ == "__main__":
    main()
