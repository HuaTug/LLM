#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 对话Demo - Azure OpenAI版本
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# 加载环境变量
load_dotenv()


class AzureChatDemo:
    def __init__(self):
        # 检查Azure OpenAI配置
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([azure_endpoint, azure_api_key, azure_deployment]):
            raise ValueError("请在.env文件中设置Azure OpenAI相关配置")

        # 初始化AzureChatOpenAI
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_key=azure_api_key,
            api_version=azure_api_version,
            temperature=0.08,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        # 设置对话记忆
        self.memory = ConversationBufferMemory()

        # 创建对话链
        template = """
你将扮演一位历史老师，与我进行对话。当我提出问题时，你要以历史老师的身份进行专业且耐心的解答。
当前对话：{history} 
这是你需要回答的问题：
{input}
在回答问题时，请遵循以下指南：
1. 使用清晰、准确、易懂的语言，避免使用过于生僻的历史术语，除非必要时进行解释。
2. 提供全面、客观的历史信息，包括事件的背景、经过、影响等。
3. 如果涉及到不同的历史观点或争议，要进行适当的说明。
4. 尽量结合具体的历史事例来支持你的观点。
5. 确保回答没有任何事实性错误。
AI:
"""

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )

        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=True
        )

    def chat(self, message):
        """发送消息并获取回复"""
        try:
            response = self.conversation.predict(input=message)
            return response
        except Exception as e:
            return f"抱歉，发生了错误: {e}"

    def direct_chat(self, message):
        """直接使用LLM聊天（不使用记忆）"""
        try:
            messages = [HumanMessage(content=message)]
            response = self.llm(messages)
            return response.content
        except Exception as e:
            return f"抱歉，发生了错误: {e}"

    def start_chat(self):
        """开始交互式聊天"""
        print("🤖 LangChain Azure OpenAI 对话Demo")
        print(f"使用部署: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
        print(f"API版本: {os.getenv('AZURE_OPENAI_API_VERSION')}")
        print("输入 'quit', 'exit' 或 'bye' 来退出聊天")
        print("输入 '/direct' 开头使用直接模式（无记忆）")
        print("-" * 50)

        while True:
            try:
                # 获取用户输入
                user_input = input("\n您: ").strip()

                # 检查退出条件
                if user_input.lower() in ['quit', 'exit', 'bye', '退出', '再见']:
                    print("AI: 再见！很高兴与您聊天！")
                    break

                if not user_input:
                    continue

                # 检查是否使用直接模式
                if user_input.startswith('/direct '):
                    message = user_input[8:]  # 去掉 '/direct ' 前缀
                    print("\nAI (直接模式): ", end="", flush=True)
                    response = self.direct_chat(message)
                    print(response)
                else:
                    # 获取AI回复（带记忆）
                    print("\nAI: ", end="", flush=True)
                    response = self.chat(user_input)
                    print(response)

            except KeyboardInterrupt:
                print("\n\nAI: 再见！很高兴与您聊天！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")


def main():
    try:
        demo = AzureChatDemo()
        demo.start_chat()
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保已创建 .env 文件并设置了Azure OpenAI相关配置：")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME")
        print("- AZURE_OPENAI_API_VERSION")
    except Exception as e:
        print(f"初始化失败: {e}")


if __name__ == "__main__":
    main()