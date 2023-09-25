default_program_schema_text = """
// A program consists of a sequence of function calls that are evaluated in order.
export type Program = {
    "@steps": FunctionCall[];
}

// A function call specifies a function name and a list of argument expressions. Arguments may contain
// nested function calls and result references.
export type FunctionCall = {
    // Name of the function
    "@func": string;
    // Arguments for the function, if any
    "@args"?: Expression[];
};

// An expression is a JSON value, a function call, or a reference to the result of a preceding expression.
export type Expression = JsonValue | FunctionCall | ResultReference;

// A JSON value is a string, a number, a boolean, null, an object, or an array. Function calls and result
// references can be nested in objects and arrays.
export type JsonValue = string | number | boolean | null | { [x: string]: Expression } | Expression[];

// A result reference represents the value of an expression from a preceding step.
export type ResultReference = {
    // Index of the previous expression in the "@steps" array
    "@ref": number;
};
`
"""


def validator_create_request_prompt(
    request, schema, program_schema_text=default_program_schema_text
):
    return (
        f"You are a service that translates user requests into programs represented as JSON using the following "
        f"TypeScript definitions:\n"
        f"```\n{program_schema_text}\n```\n"
        f"The programs can call functions from the API defined in the following TypeScript definitions:\n\n"
        f'"""\n{schema}\n"""\n'
        f"The following is a user request:\n"
        f'"""\n{request}\n"""\n'
        f"The following is the user request translated into a JSON program object with 2 spaces of indentation and "
        f"no properties with the value undefined:\n"
    )
