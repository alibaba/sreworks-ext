def generator_create_request_prompt(request, schema, type_name):
    return (
        f'You are a service that translates user requests into JSON objects of type "{type_name}" according to the following TypeScript definitions:\n'
        f"```\n{schema}\n```\n"
        f"Focus on what is marked by ~ ~"
        f"The following is a user request:\n"
        f'"""\n{request}\n"""\n'
        f"The following is the user request translated into a JSON object with 2 spaces of indentation and no "
        f"properties with the value undefined:\n"
    )
