from setuptools import setup, find_packages


setup(
    name="agitops",  # 你的包名，例如 "my_awesome_library"
    version="0.0.11",  # 包的版本号，遵循语义化版本规范 (例如 0.1.0, 1.0.0, 2.3.5)
    author="twwyzh",  # 你的名字
    author_email="twwyzh@gmail.com",  # 你的邮箱地址
    description="gitops for ai brain",  # 包的简短描述
    long_description="gitops for ai brain",  # 包的详细描述，通常来自 README 文件
    long_description_content_type="text/markdown",  # 长描述的格式，这里是 Markdown
    url="https://github.com/alibaba/sreworks-ext/tree/master/agitops",  # 项目的 URL，例如 GitHub 仓库地址
    packages=find_packages(include=["agitops", "agitops.*"]),
    package_dir={"": "."},  # 告诉 setuptools 包在 'src' 目录下
                             # 如果你的代码直接在根目录，可以省略这一行
    # 如果你的包包含非 Python 文件 (如数据文件、模板文件)，使用 include_package_data 和 package_data
    # include_package_data=True,
    # package_data={
    #     # "package_name": ["data/*.dat"], # 示例：包含 package_name/data 目录下的 .dat 文件
    # },
    classifiers=[  # 包的分类信息，帮助用户在 PyPI 上找到你的包
        "Development Status :: 3 - Alpha",  # 开发阶段：3 - Alpha, 4 - Beta, 5 - Production/Stable
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Build Tools",
        "License :: OSI Approved :: GNU General Public License v2 (GPLv2)",  # 你选择的开源许可证
        "Programming Language :: Python :: 3",  # 支持的 Python 版本
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",  # 操作系统兼容性
    ],
    python_requires=">=3.7",  # 要求的最低 Python 版本
    install_requires=[  # 项目依赖的包，会在安装你的包时自动安装
        "openai",
        "pyyaml",
    ],
    extras_require={  # 可选依赖，用户可以根据需要安装
        # "dev": ["pytest>=3.7", "flake8", "black"], # 示例：开发环境依赖
        # "docs": ["sphinx", "sphinx-rtd-theme"], # 示例：文档生成依赖
    },
    project_urls={  # 项目相关的其他链接
        "Source Code": "https://github.com/alibaba/sreworks-ext/tree/master/agitops",
    },
    keywords="sample, setuptools, development", # 描述你的包的关键词
    # license="GPL-2.0", # 许可证名称，如果已在 classifiers 中指定，这里可以省略
)