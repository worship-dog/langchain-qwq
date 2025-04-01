<picture>
  <source media="(prefers-color-scheme: light)" srcset="docs/static/img/logo-dark.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/static/img/logo-light.svg">
  <img alt="LangChain Logo" src="docs/static/img/logo-dark.svg" width="80%">
</picture>

<div>
<br>
</div>

[![PyPI - License](https://img.shields.io/pypi/l/langchain-core?style=flat-square)](https://opensource.org/licenses/MIT)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/langchain-core?style=flat-square)](https://pypistats.org/packages/langchain-qwq)
[![GitHub star chart](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat-square)](https://star-history.com/#yigit353/langchain-qwq)
[![Open Issues](https://img.shields.io/github/issues-raw/langchain-ai/langchain?style=flat-square)](https://github.com/yigit353/langchain-qwq/issues)
[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=visualstudiocode&style=flat-square)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/yigit353/langchain-qwq)
[<img src="https://github.com/codespaces/badge.svg" title="Open in Github Codespace" width="150" height="20">](https://codespaces.new/yigit353/langchain-qwq)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/langchainai.svg?style=social&label=Follow%20%40LangChainAI)](https://x.com/yigitbekir)

# langchain-qwq

This package contains the LangChain integration with QwQ

## Installation

```bash
pip install -U langchain-qwq
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatQwQ` class exposes chat models from QwQ.

```python
from langchain_qwq import ChatQwQ

llm = ChatQwQ()
llm.invoke("Sing a ballad of LangChain.")
```
