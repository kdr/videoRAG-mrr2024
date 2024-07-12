import json
import logging
import os
import sys
from typing import List

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors.llm_selectors import LLMSingleSelector, LLMMultiSelector
from llama_index.core.tools.query_engine import QueryEngineTool
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.milvus import MilvusVectorStore
from pydantic import BaseModel, Field
import streamlit as st

# ===== Query Engine Setup ===== #

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

HOWTO_STORAGE = "./storage/howto-212k-bge-small-en-384"
HOWTO_COLLECTION = "howto212k"
TRAVEL_STORAGE = "./storage/travel-69k-bge-small-en-384"
TRAVEL_COLLECTION = "travel69k"

EMBED = "BAAI/bge-small-en-v1.5"
EMBED_DIMS = 384
RESPONSE_MODE = "tree_summarize"


class HowToStep(BaseModel):
    """How to step data model"""

    description: str = Field(
        description=
        "description of specific step to take in series of steps to solve problem / accomplish task"
    )
    url: str = Field(description="link to the source youtube video")
    scene: int = Field(
        description=
        "relevant video scene number in source video, only provide an integer number"
    )
    timestamp: int = Field(
        description=
        "relevant timestamp of scene source video, only provide an integer number"
    )


class HowToSteps(BaseModel):
    """Data model for how to step extracted information."""

    steps: List[HowToStep] = Field(
        description="List of steps to take to solve problem / accomplish task")
    summary: str = Field(
        description=
        "High level summary / commentary on how to solve problem / accomplish. task"
    )


class NotablePlaceMention(BaseModel):
    """Notable place mention data model"""

    name: str = Field(description="the place name")
    description: str = Field(
        description=
        "one to two sentence description of the place and why it is well known / why people love it"
    )
    place_type: str = Field(
        description=
        "type of place / point of interest, e.g. restaurant, landmark, transportation, shopping, etc."
    )
    best_known_for: List[str] = Field(
        description="list of things this place is know for")
    url: str = Field(description="link to the source youtube video")
    scene: int = Field(
        description=
        "relevant video scene number in source video, only provide an integer number"
    )
    timestamp: int = Field(
        description=
        "relevant timestamp of scene source video, only provide an integer number"
    )


class NotablePlaceMentionsSummary(BaseModel):
    """Data model for notable place mentions extracted information."""

    place_mentions: List[NotablePlaceMention] = Field(
        description="List of notable places mentioned in retrieved results")
    summary: str = Field(
        description=
        "High level summary / commentary on the places retrieved and how relevant to the query"
    )


def get_milvus_query_engine(collection_name,
                            storage_dir,
                            output_cls,
                            llm,
                            response_mode=RESPONSE_MODE):
    vector_store = MilvusVectorStore(dim=EMBED_DIMS,
                                     uri=os.getenv('MILVUS_HOST'),
                                     token=os.getenv('MILVUS_TOKEN'),
                                     overwrite=False,
                                     collection_name=collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store,
                                                   persist_dir=storage_dir)
    index = load_index_from_storage(storage_context=storage_context)

    query_engine = index.as_query_engine(
        output_cls=output_cls,
        response_mode=response_mode,
        llm=llm,
        verbose=True,
    )

    return query_engine


@st.cache_resource
def load_query_engine():
    embed_model = HuggingFaceEmbedding(model_name=EMBED)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=1)

    from llama_index.core import Settings
    Settings.embed_model = embed_model
    Settings.llm = llm

    howto_query_engine = get_milvus_query_engine(HOWTO_COLLECTION,
                                                 HOWTO_STORAGE, HowToSteps,
                                                 llm)
    place_query_engine = get_milvus_query_engine(TRAVEL_COLLECTION,
                                                 TRAVEL_STORAGE,
                                                 NotablePlaceMentionsSummary,
                                                 llm)

    howto_tool = QueryEngineTool.from_defaults(
        query_engine=howto_query_engine,
        description=
        "Useful for retrieving information about how to accomplish specific tasks",
    )
    place_tool = QueryEngineTool.from_defaults(
        query_engine=place_query_engine,
        description=
        "Useful for retrieving information about notable places like restaurants, tourist attractions, shopping, landmarks, etc.",
    )

    # Create Router Query Engine
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            place_tool,
            howto_tool,
        ],
    )

    return query_engine


query_engine = load_query_engine()

# =====  Start Main Stream Lit App ===== #

with st.sidebar:
    st.title("Thousand Words Video Explorer")
    st.markdown(f"""
Loaded Indices: `How-To, Travel`

""")
    st.markdown("""
Info:

* Index contains ~282.1K YouTube videos that appeared to be english language and were in the "How To & Style" or "Travel & Events" youtube category
* This simple video DB is representing over 12.4k hours of video (i.e. ~1.42 years of audio/visual information)

Example Queries:

* `what is the best museum in paris aside from the lourve?`
* `how do I repair a loose toilet flusher?`
* `what are the top 5 places in england to visit for a harry potter fan?`
* `show me the best steakhouses in new york city`
* `show me step by step how to make mapu tofu`

""")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            response = message["content"]
            print(response)
            print(type(response.response))
            st.text("Here's what I found:")
            if str(type(response.response)) == "<class '__main__.HowToSteps'>":
                st.markdown(response.response.summary)
                for no, m in enumerate(response.response.steps):
                    st.header(f'Step {no}')
                    st.markdown(m.description)
                    if m.url:
                        split = m.url[len('https://www.youtube.com/watch?v='
                                          ):] + '_split_' + f'{m.scene:05d}'
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(
                                f"https://storage.googleapis.com/kdr-public/pandas70m/howto-travel/img/clip-start/{split}.jpg"
                            )
                            st.text("clip preview")
                        with col2:
                            st.video(m.url,
                                     format="video/mp4",
                                     start_time=m.timestamp)
            elif str(type(response.response)
                     ) == "<class '__main__.NotablePlaceMentionsSummary'>":
                st.markdown(response.response.summary)
                for m in response.response.place_mentions:
                    st.header(m.name)
                    t = m.description + "\n\n**Known For**:\n"
                    for b in m.best_known_for:
                        t += "\n* " + b
                    st.markdown(t)
                    if m.url:
                        split = m.url[len('https://www.youtube.com/watch?v='
                                          ):] + '_split_' + f'{m.scene:05d}'
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(
                                f"https://storage.googleapis.com/kdr-public/pandas70m/howto-travel/img/clip-start/{split}.jpg"
                            )
                            st.text("clip preview")
                        with col2:
                            st.video(m.url,
                                     format="video/mp4",
                                     start_time=m.timestamp)

            st.divider()
            vs = set([])
            t = "Sources:\n\n"
            for s in response.source_nodes:
                if s.metadata['video_url'] not in vs:
                    t += f"1. [{s.metadata['video_title']}]({s.metadata['video_url']})\n"
                    vs.add(s.metadata['video_url'])
            st.markdown(t)

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = query_engine.query(prompt)
        print(response)
        print(type(response.response))
        st.text("Here's what I found:")
        if str(type(response.response)) == "<class '__main__.HowToSteps'>":
            st.markdown(response.response.summary)
            for no, m in enumerate(response.response.steps):
                st.header(f'Step {no}')
                st.markdown(m.description)
                if m.url:
                    split = m.url[len('https://www.youtube.com/watch?v='
                                      ):] + '_split_' + f'{m.scene:05d}'
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            f"https://storage.googleapis.com/kdr-public/pandas70m/howto-travel/img/clip-start/{split}.jpg"
                        )
                        st.text("clip preview")
                    with col2:
                        st.video(m.url,
                                 format="video/mp4",
                                 start_time=m.timestamp)
        elif str(type(response.response)
                 ) == "<class '__main__.NotablePlaceMentionsSummary'>":
            st.markdown(response.response.summary)
            for m in response.response.place_mentions:
                st.header(m.name)
                t = m.description + "\n\n**Known For**:\n"
                for b in m.best_known_for:
                    t += "\n* " + b
                st.markdown(t)
                if m.url:
                    split = m.url[len('https://www.youtube.com/watch?v='
                                      ):] + '_split_' + f'{m.scene:05d}'
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(
                            f"https://storage.googleapis.com/kdr-public/pandas70m/howto-travel/img/clip-start/{split}.jpg"
                        )
                        st.text("clip preview")
                    with col2:
                        st.video(m.url,
                                 format="video/mp4",
                                 start_time=m.timestamp)
        st.divider()
        vs = set([])
        t = "Sources:\n\n"
        for s in response.source_nodes:
            if s.metadata['video_url'] not in vs:
                t += f"1. [{s.metadata['video_title']}]({s.metadata['video_url']})\n"
                vs.add(s.metadata['video_url'])
        st.markdown(t)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
