import openai
import yaml
import os
import json
import validator
import evaluator
import generator


def read_config_from_yaml(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


config = read_config_from_yaml("config.yaml")

openai.api_key = config["openai"]["OPENAI_API_KEY"]
openai.api_base = config["openai"]["OPENAI_API_BASE"]
openai_model = config["openai"]["OPENAI_MODEL"]


def read_request_from_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            request = file.read()
            return request
    except FileNotFoundError:
        print("FileNotFoundError in read_request_from_file")


def read_type_from_file(file_name):
    file_path = os.path.join(os.path.dirname(__file__), file_name)
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            schema = file.read()
            return schema
    except FileNotFoundError:
        print("FileNotFoundError in read_type_from_file")


# 这是生成器
completion = openai.ChatCompletion.create(
    model=openai_model,
    messages=[
        {
            "role": "user",
            "content": generator.generator_create_request_prompt(
                read_request_from_file("input.txt"),
                read_type_from_file("type/WelcomeCard.ts"),
                "WelcomeCard",
            ),
        }
    ],
)

generator_completion = completion.choices[0].message.content

try:
    program = json.loads(generator_completion)
    print(program)
except:
    print("error json decode")

# # 这是typechat的验证器
# completion2 = openai.ChatCompletion.create(
#     model=openai_model,
#     messages=[
#         {
#             "role": "user",
#             "content": validator.validator_create_request_prompt(
#                 read_request_from_file("input.txt"),
#                 read_type_from_file("type/Table.ts"),
#             ),
#         }
#     ],
# )
# validator_completion = completion2.choices[0].message.content
# # json验证器加载检查
# try:
#     program = json.loads(validator_completion)
#     func_evaluator = evaluator.Evaluator()
#     func = asyncio.run(func_evaluator.evaluate_json_program(program))
# except:
#     print("error json decode")
