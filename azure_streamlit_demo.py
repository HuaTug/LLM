#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangChain 对话Demo - Azure OpenAI Streamlit Web界面版本
运行命令: streamlit run azure_streamlit_demo.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

# 加载环境变量
load_dotenv()

# 设置页面配置
st.set_page_config(
    page_title="LangChain Azure OpenAI 对话Demo",
    page_icon="🤖",
    layout="wide"
)

# 初始化session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation" not in st.session_state:
    st.session_state.conversation = None


def init_conversation():
    """初始化对话链"""
    try:
        # 检查Azure OpenAI配置
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        azure_api_version = os.getenv("AZURE_OPENAI_API_VERSION")

        if not all([azure_endpoint, azure_api_key, azure_deployment]):
            st.error("请在.env文件中设置Azure OpenAI相关配置")
            return None

        # 初始化AzureChatOpenAI
        llm = AzureChatOpenAI(
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
        memory = ConversationBufferMemory()

        # 创建对话链
        template = """以下是人类和AI之间的友好对话。AI是健谈的，并提供了很多具体的细节。如果AI不知道问题的答案，它会诚实地说它不知道。

当前对话:
{history}
人类: {input}
AI:"""

        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template=template
        )

        conversation = ConversationChain(
            llm=llm,
            memory=memory,
            prompt=prompt,
            verbose=False
        )

        return conversation
    except Exception as e:
        st.error(f"初始化失败: {e}")
        return None


def main():
    st.title("🤖 LangChain Azure OpenAI 对话Demo")

    # 显示配置信息
    if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
        st.info(
            f"🔧 当前部署: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')} | API版本: {os.getenv('AZURE_OPENAI_API_VERSION')}")

    st.markdown("---")

    # 初始化对话链
    if st.session_state.conversation is None:
        st.session_state.conversation = init_conversation()

    if st.session_state.conversation is None:
        st.stop()

    # 侧边栏设置
    with st.sidebar:
        st.header("⚙️ 设置")

        # 温度控制
        temperature = st.slider(
            "温度 (Temperature)",
            min_value=0.0,
            max_value=2.0,
            value=0.08,
            step=0.01,
            help="控制回复的随机性，越高越有创意"
        )

        # 更新温度设置
        if hasattr(st.session_state.conversation.llm, 'temperature'):
            st.session_state.conversation.llm.temperature = temperature

        if st.button("🗑️ 清除对话历史"):
            st.session_state.messages = []
            st.session_state.conversation.memory.clear()
            st.rerun()

        st.markdown("---")
        st.markdown("### 📖 使用说明")
        st.markdown("""
        1. 在聊天框中输入您的问题
        2. 按Enter或点击发送按钮
        3. AI会根据上下文回复您
        4. 可以调整温度控制回复风格
        5. 点击"清除对话历史"重新开始
        """)

        st.markdown("---")
        st.markdown("### ℹ️ 配置信息")
        if os.getenv("AZURE_OPENAI_ENDPOINT"):
            st.text(f"端点: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
        if os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"):
            st.text(f"部署: {os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')}")
        if os.getenv("AZURE_OPENAI_API_VERSION"):
            st.text(f"API版本: {os.getenv('AZURE_OPENAI_API_VERSION')}")

    # 显示聊天历史
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # 聊天输入
    if prompt := st.chat_input("请输入您的问题..."):
        # 添加用户消息到历史
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 获取AI回复
        with st.chat_message("assistant"):
            with st.spinner("AI正在思考..."):
                try:
                    response = st.session_state.conversation.predict(input=prompt)
                    st.markdown(response)

                    # 添加AI回复到历史
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"抱歉，发生了错误: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()