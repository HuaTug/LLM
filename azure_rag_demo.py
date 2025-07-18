#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于Azure OpenAI的RAG (检索增强生成) Demo
实现获取最新消息并提供给大模型的功能
"""

import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain imports
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 加载环境变量
load_dotenv()


class MessageRetriever:
    """消息检索器 - 模拟从各种数据源获取最新消息"""

    def __init__(self):
        # 这里模拟一些数据源，实际应用中可以是：
        # - 数据库查询
        # - API调用
        # - 文件读取
        # - 实时数据流等
        self.data_sources = {
            "news": self._get_latest_news,
            "company_updates": self._get_company_updates,
            "market_data": self._get_market_data,
            "user_messages": self._get_user_messages
        }

    def _get_latest_news(self) -> List[Dict[str, Any]]:
        """获取最新新闻（模拟）"""
        # 实际实现中，这里可以调用新闻API
        return [
            {
                "title": "人工智能技术突破",
                "content": "最新的AI模型在多个基准测试中创下新纪录，显示出强大的推理能力。",
                "timestamp": datetime.now() - timedelta(hours=2),
                "source": "tech_news",
                "category": "technology"
            },
            {
                "title": "云计算市场增长",
                "content": "2024年云计算市场预计将增长25%，主要驱动力来自AI和数据分析需求。",
                "timestamp": datetime.now() - timedelta(hours=1),
                "source": "business_news",
                "category": "business"
            }
        ]

    def _get_company_updates(self) -> List[Dict[str, Any]]:
        """获取公司更新（模拟）"""
        return [
            {
                "title": "新产品发布",
                "content": "公司将于下月发布基于AI的新一代客户服务平台，预计将提高效率30%。",
                "timestamp": datetime.now() - timedelta(hours=3),
                "source": "internal",
                "category": "product"
            }
        ]

    def _get_market_data(self) -> List[Dict[str, Any]]:
        """获取市场数据（模拟）"""
        return [
            {
                "title": "股市动态",
                "content": "科技股今日上涨2.5%，AI相关公司表现突出。投资者对新技术前景保持乐观。",
                "timestamp": datetime.now() - timedelta(minutes=30),
                "source": "market_feed",
                "category": "finance"
            }
        ]

    def _get_user_messages(self) -> List[Dict[str, Any]]:
        """获取用户消息（模拟）"""
        return [
            {
                "title": "用户反馈",
                "content": "用户反映新版本界面更加友好，但希望增加更多自定义选项。",
                "timestamp": datetime.now() - timedelta(hours=4),
                "source": "user_feedback",
                "category": "feedback"
            }
        ]

    def get_latest_messages(self, hours_back: int = 24, categories: List[str] = None) -> List[Dict[str, Any]]:
        """获取最新消息"""
        all_messages = []
        cutoff_time = datetime.now() - timedelta(hours=hours_back)

        for source_name, source_func in self.data_sources.items():
            messages = source_func()
            for msg in messages:
                if msg["timestamp"] >= cutoff_time:
                    if categories is None or msg["category"] in categories:
                        all_messages.append(msg)

        # 按时间排序，最新的在前
        all_messages.sort(key=lambda x: x["timestamp"], reverse=True)
        return all_messages


class AzureRAGDemo:
    """基于Azure OpenAI的RAG演示类"""

    def __init__(self):
        # 检查Azure OpenAI配置
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        print(f"初始化Azure OpenAI组件...", azure_endpoint, azure_api_key, azure_deployment, azure_api_version)
        if not all([azure_endpoint, azure_api_key, azure_deployment]):
            raise ValueError("请在.env文件中设置Azure OpenAI相关配置")

        # 初始化Azure OpenAI组件
        self.llm = AzureChatOpenAI(
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
            api_key=azure_api_key,
            api_version=azure_api_version,
            temperature=0.1,
            max_tokens=None,
        )

        # 初始化嵌入模型（需要单独的嵌入模型部署）
        # 如果没有嵌入模型部署，可以使用OpenAI的嵌入API
        try:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002"),
                api_key=azure_api_key,
                api_version=azure_api_version,
            )
        except:
            print("警告: 无法初始化Azure嵌入模型，将使用OpenAI嵌入模型")
            from langchain_openai import OpenAIEmbeddings
            self.embeddings = OpenAIEmbeddings()

        # 初始化消息检索器
        self.message_retriever = MessageRetriever()

        # 初始化向量存储
        self.vector_store = None
        self.retriever = None

        # 设置RAG链
        self._setup_rag_chain()

    def _setup_rag_chain(self):
        """设置RAG链"""
        # 创建RAG提示模板
        rag_prompt = ChatPromptTemplate.from_template("""
你是一个智能助手，专门用于回答基于最新信息的问题。

请基于以下提供的上下文信息来回答用户的问题：

上下文信息:
{context}

用户问题: {input}

请注意：
1. 请主要基于提供的上下文信息来回答
2. 如果上下文中没有相关信息，请明确说明
3. 回答要准确、简洁、有用
4. 如果可能，请提及信息的时间和来源

回答:
""")

        # 创建文档组合链
        self.combine_docs_chain = create_stuff_documents_chain(
            self.llm, rag_prompt
        )

    def update_knowledge_base(self, hours_back: int = 24, categories: List[str] = None):
        """更新知识库with最新消息"""
        print(f"正在获取过去{hours_back}小时的最新消息...")

        # 获取最新消息
        latest_messages = self.message_retriever.get_latest_messages(
            hours_back=hours_back,
            categories=categories
        )

        if not latest_messages:
            print("没有找到最新消息")
            return

        print(f"找到 {len(latest_messages)} 条最新消息")

        # 将消息转换为文档
        documents = []
        for msg in latest_messages:
            # 创建包含元数据的文档
            doc_content = f"标题: {msg['title']}\n内容: {msg['content']}"
            metadata = {
                "title": msg["title"],
                "timestamp": msg["timestamp"].isoformat(),
                "source": msg["source"],
                "category": msg["category"]
            }
            documents.append(Document(page_content=doc_content, metadata=metadata))

        # 文本分割（如果消息很长）
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            add_start_index=True,
        )
        splits = text_splitter.split_documents(documents)

        # 创建或更新向量存储
        print("正在构建向量存储...")
        self.vector_store = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            collection_name="latest_messages"
        )

        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # 返回最相关的3个文档
        )

        print("知识库更新完成！")

    def chat_with_latest_info(self, question: str) -> str:
        """基于最新信息进行对话"""
        if self.retriever is None:
            return "请先调用 update_knowledge_base() 方法来构建知识库"

        # 创建检索链
        retrieval_chain = create_retrieval_chain(
            self.retriever,
            self.combine_docs_chain
        )

        # 执行检索和生成
        result = retrieval_chain.invoke({"input": question})

        return result["answer"]

    def get_relevant_context(self, question: str, k: int = 3) -> List[Document]:
        """获取相关上下文文档"""
        if self.retriever is None:
            return []

        return self.retriever.invoke(question)

    def start_interactive_chat(self):
        """开始交互式聊天"""
        print("🤖 Azure OpenAI RAG 演示")
        print("基于最新消息的智能问答系统")
        print("-" * 50)

        # 首先更新知识库
        print("初始化知识库...")
        self.update_knowledge_base()
        print()

        print("可以开始提问了！")
        print("输入 'quit', 'exit' 或 'bye' 来退出")
        print("输入 'update' 来更新知识库")
        print("输入 'context <问题>' 来查看相关上下文")
        print("-" * 50)

        while True:
            try:
                user_input = input("\n您: ").strip()

                if user_input.lower() in ['quit', 'exit', 'bye', '退出']:
                    print("AI: 再见！")
                    break

                if not user_input:
                    continue

                if user_input.lower() == 'update':
                    print("更新知识库...")
                    self.update_knowledge_base()
                    print("知识库已更新！")
                    continue

                if user_input.lower().startswith('context '):
                    question = user_input[8:]
                    docs = self.get_relevant_context(question)
                    print(f"\n找到 {len(docs)} 个相关文档:")
                    for i, doc in enumerate(docs, 1):
                        print(f"\n文档 {i}:")
                        print(f"内容: {doc.page_content[:200]}...")
                        print(f"元数据: {doc.metadata}")
                    continue

                # 正常问答
                print("\nAI: ", end="", flush=True)
                response = self.chat_with_latest_info(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nAI: 再见！")
                break
            except Exception as e:
                print(f"\n发生错误: {e}")


def main():
    """主函数"""
    try:
        demo = AzureRAGDemo()
        demo.start_interactive_chat()
    except ValueError as e:
        print(f"配置错误: {e}")
        print("\n请确保已创建 .env 文件并设置了以下配置：")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_API_KEY")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME")
        print("- AZURE_OPENAI_API_VERSION")
        print("- AZURE_OPENAI_EMBEDDING_DEPLOYMENT (可选)")
    except Exception as e:
        print(f"初始化失败: {e}")


if __name__ == "__main__":
    main()