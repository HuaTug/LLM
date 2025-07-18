#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 高级对话Demo
包含角色设定、对话历史管理、多种记忆类型等功能
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
import json
from datetime import datetime

# 加载环境变量
load_dotenv()


class AdvancedChatDemo:
    def __init__(self, persona="helpful_assistant"):
        # 检查API密钥
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("请在.env文件中设置OPENAI_API_KEY")

        # 初始化ChatOpenAI
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        )

        # 设置角色模板
        self.personas = {
            "helpful_assistant": {
                "name": "智能助手",
                "template": """你是一个友好、有帮助的AI助手。你很乐意回答问题并提供有用的信息。你的回答要准确、详细，同时保持友好的语调。

当前对话:
{history}
用户: {input}
助手:"""
            },
            "creative_writer": {
                "name": "创意作家",
                "template": """你是一位富有创意的作家和storyteller。你擅长创作故事、诗歌，并能够用生动有趣的方式表达想法。你的回答要富有想象力和创造性。

当前对话:
{history}
用户: {input}
作家:"""
            },
            "technical_expert": {
                "name": "技术专家",
                "template": """你是一位经验丰富的技术专家，精通编程、软件开发和技术架构。你的回答要专业、准确，并且能够提供具体的技术解决方案。

当前对话:
{history}
用户: {input}
专家:"""
            },
            "teacher": {
                "name": "老师",
                "template": """你是一位耐心的老师，擅长解释复杂的概念。你会循序渐进地教授知识，使用简单易懂的语言，并且经常举例来帮助理解。

当前对话:
{history}
学生: {input}
老师:"""
            }
        }

        self.current_persona = persona
        self.setup_conversation()

    def setup_conversation(self):
        """设置对话链"""
        # 使用摘要缓冲记忆，可以处理更长的对话
        self.memory = ConversationSummaryBufferMemory(
            llm=self.llm,
            max_token_limit=1000,
            return_messages=True
        )

        # 获取当前角色的模板
        persona_info = self.personas[self.current_persona]
        template = persona_info["template"]

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )

        self.conversation = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=False
        )

    def switch_persona(self, persona):
        """切换角色"""
        if persona in self.personas:
            self.current_persona = persona
            self.setup_conversation()
            return f"已切换到角色: {self.personas[persona]['name']}"
        else:
            return f"未知角色: {persona}。可用角色: {', '.join(self.personas.keys())}"

    def chat(self, message):
        """发送消息并获取回复"""
        try:
            response = self.conversation.predict(input=message)
            return response
        except Exception as e:
            return f"抱歉，发生了错误: {e}"

    def save_conversation(self, filename=None):
        """保存对话历史"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"conversation_{timestamp}.json"

        try:
            # 获取对话历史
            messages = []
            for message in self.memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    messages.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    messages.append({"role": "assistant", "content": message.content})

            conversation_data = {
                "persona": self.current_persona,
                "timestamp": datetime.now().isoformat(),
                "messages": messages
            }

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)

            return f"对话已保存到: {filename}"
        except Exception as e:
            return f"保存失败: {e}"

    def load_conversation(self, filename):
        """加载对话历史"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                conversation_data = json.load(f)

            # 切换到保存时的角色
            self.switch_persona(conversation_data["persona"])

            # 恢复对话历史
            for message in conversation_data["messages"]:
                if message["role"] == "user":
                    self.memory.chat_memory.add_user_message(message["content"])
                elif message["role"] == "assistant":
                    self.memory.chat_memory.add_ai_message(message["content"])

            return f"已加载对话: {filename}"
        except Exception as e:
            return f"加载失败: {e}"

    def show_memory_stats(self):
        """显示记忆统计信息"""
        total_messages = len(self.memory.chat_memory.messages)
        buffer_string = self.memory.buffer
        return f"对话消息数: {total_messages}\n当前缓冲区内容长度: {len(buffer_string)} 字符"

    def start_chat(self):
        """开始交互式聊天"""
        print("🤖 LangChain 高级对话Demo")
        print(f"当前角色: {self.personas[self.current_persona]['name']}")
        print("\n可用命令:")
        print("  /persona <角色名> - 切换角色")
        print("  /personas - 显示所有可用角色")
        print("  /save [文件名] - 保存对话")
        print("  /load <文件名> - 加载对话")
        print("  /memory - 显示记忆统计")
        print("  /clear - 清除对话历史")
        print("  /help - 显示帮助")
        print("  /quit - 退出")
        print("-" * 60)

        while True:
            try:
                # 获取用户输入
                user_input = input(f"\n您: ").strip()

                # 处理命令
                if user_input.startswith('/'):
                    if user_input == '/quit':
                        print("AI: 再见！很高兴与您聊天！")
                        break
                    elif user_input == '/help':
                        print("可用命令:")
                        print("  /persona <角色名> - 切换角色")
                        print("  /personas - 显示所有可用角色")
                        print("  /save [文件名] - 保存对话")
                        print("  /load <文件名> - 加载对话")
                        print("  /memory - 显示记忆统计")
                        print("  /clear - 清除对话历史")
                        print("  /help - 显示帮助")
                        print("  /quit - 退出")
                        continue
                    elif user_input == '/personas':
                        print("可用角色:")
                        for key, value in self.personas.items():
                            current = " (当前)" if key == self.current_persona else ""
                            print(f"  {key}: {value['name']}{current}")
                        continue
                    elif user_input.startswith('/persona '):
                        persona = user_input[9:].strip()
                        result = self.switch_persona(persona)
                        print(f"系统: {result}")
                        continue
                    elif user_input.startswith('/save'):
                        parts = user_input.split(' ', 1)
                        filename = parts[1] if len(parts) > 1 else None
                        result = self.save_conversation(filename)
                        print(f"系统: {result}")
                        continue
                    elif user_input.startswith('/load '):
                        filename = user_input[6:].strip()
                        result = self.load_conversation(filename)
                        print(f"系统: {result}")
                        continue
                    elif user_input == '/memory':
                        stats = self.show_memory_stats()
                        print(f"系统: {stats}")
                        continue
                    elif user_input == '/clear':
                        self.memory.clear()
                        print("系统: 对话历史已清除")
                        continue
                    else:
                        print("系统: 未知命令，输入 /help 查看帮助")
                        continue

                if not user_input:
                    continue

                # 获取AI回复
                current_persona_name = self.personas[self.current_persona]['name']
                print(f"\n{current_persona_name}: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)

            except KeyboardInterrupt:
                print(f"\n\n{self.personas[self.current_persona]['name']}: 再见！很高兴与您聊天！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")


def main():
    try:
        demo = AdvancedChatDemo()
        demo.start_chat()
    except ValueError as e:
        print(f"配置错误: {e}")
        print("请确保已创建 .env 文件并设置了 OPENAI_API_KEY")
    except Exception as e:
        print(f"初始化失败: {e}")


if __name__ == "__main__":
    main()