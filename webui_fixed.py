import pdb
import logging

from dotenv import load_dotenv

load_dotenv()
import os
import glob
import asyncio
import argparse
import os

logger = logging.getLogger(__name__)

import gradio as gr
# Náhrada za gradio_i18n
import json
import os

class SimpleTranslate:
    def __init__(self, languages=None, default_language="en", locales_dir="locales"):
        self.languages = languages or ["en"]
        self.default_language = default_language
        self.locales_dir = locales_dir
        self.current_language = default_language
        self.translations = {}

        # Načtení překladů
        for lang in self.languages:
            try:
                with open(os.path.join(self.locales_dir, f"{lang}.json"), "r", encoding="utf-8") as f:
                    self.translations[lang] = json.load(f)
            except Exception as e:
                print(f"Chyba při načítání překladu pro {lang}: {e}")
                self.translations[lang] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def __call__(self, key):
        # Rozdělení klíče podle teček
        parts = key.split(".")

        # Získání překladu
        translation = self.translations.get(self.current_language, {})
        for part in parts:
            translation = translation.get(part, None)
            if translation is None:
                # Pokud překlad neexistuje, vrátíme klíč
                return key

        return translation

    def set_language(self, language):
        if language in self.languages:
            self.current_language = language

# Použití SimpleTranslate místo Translate
Translate = SimpleTranslate

from browser_use.agent.service import Agent
from playwright.async_api import async_playwright
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextConfig,
    BrowserContextWindowSize,
)
from langchain_ollama import ChatOllama
from playwright.async_api import async_playwright
from src.utils.agent_state import AgentState

from src.utils import utils
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser
from src.agent.custom_prompts import CustomSystemPrompt, CustomAgentMessagePrompt
from src.browser.custom_context import BrowserContextConfig, CustomBrowserContext
from src.controller.custom_controller import CustomController
from gradio.themes import Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft, Base
from src.utils.default_config_settings import default_config, load_config_from_file, save_config_to_file, \
    save_current_config, update_ui_from_config
from src.utils.utils import update_model_dropdown, get_latest_files, capture_screenshot

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None

# Create the global agent state instance
_global_agent_state = AgentState()


def resolve_sensitive_env_variables(text):
    """
    Replace environment variable placeholders ($SENSITIVE_*) with their values.
    Only replaces variables that start with SENSITIVE_.
    """
    if not text:
        return text

    import re

    # Find all $SENSITIVE_* patterns
    env_vars = re.findall(r'\$SENSITIVE_[A-Za-z0-9_]*', text)

    result = text
    for var in env_vars:
        # Remove the $ prefix to get the actual environment variable name
        env_name = var[1:]  # removes the $
        env_value = os.getenv(env_name)
        if env_value is not None:
            # Replace $SENSITIVE_VAR_NAME with its value
            result = result.replace(var, env_value)

    return result


def stop_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent

    try:
        if _global_agent is not None:
            # Request stop
            _global_agent.stop()
        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


def stop_research_agent():
    """Request the agent to stop and update UI with enhanced feedback"""
    global _global_agent_state

    try:
        # Request stop
        _global_agent_state.request_stop()

        # Update UI immediately
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        # Return UI updates
        return (  # errors_output
            gr.update(value="Stopping...", interactive=False),  # stop_button
            gr.update(interactive=False),  # run_button
        )
    except Exception as e:
        error_msg = f"Error during stop: {str(e)}"
        logger.error(error_msg)
        return (
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True)
        )


async def run_browser_agent(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    # Převod českých názvů na původní hodnoty
    if agent_type == "standardní":
        agent_type = "org"
    elif agent_type == "vlastní":
        agent_type = "custom"
    try:
        # Disable recording if the checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure the recording directory exists if recording is enabled
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        # Get the list of existing videos before the agent runs
        existing_videos = set()
        if save_recording_path:
            existing_videos = set(
                glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
                + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
            )

        task = resolve_sensitive_env_variables(task)

        # Run the agent
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )
        if agent_type == "org":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_org_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        elif agent_type == "custom":
            final_result, errors, model_actions, model_thoughts, trace_file, history_file = await run_custom_agent(
                llm=llm,
                use_own_browser=use_own_browser,
                keep_browser_open=keep_browser_open,
                headless=headless,
                disable_security=disable_security,
                window_w=window_w,
                window_h=window_h,
                save_recording_path=save_recording_path,
                save_agent_history_path=save_agent_history_path,
                save_trace_path=save_trace_path,
                task=task,
                add_infos=add_infos,
                max_steps=max_steps,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                chrome_cdp=chrome_cdp,
                max_input_tokens=max_input_tokens
            )
        else:
            raise ValueError(f"Invalid agent type: {agent_type}")

        # Get the list of videos after the agent runs (if recording is enabled)
        # latest_video = None
        # if save_recording_path:
        #     new_videos = set(
        #         glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4"))
        #         + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))
        #     )
        #     if new_videos - existing_videos:
        #         latest_video = list(new_videos - existing_videos)[0]  # Get the first new video

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            gif_path,
            trace_file,
            history_file,
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )

    except gr.Error:
        raise

    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return (
            '',  # final_result
            errors,  # errors
            '',  # model_actions
            '',  # model_thoughts
            None,  # latest_video
            None,  # history_file
            None,  # trace_file
            gr.update(value="Stop", interactive=True),  # Re-enable stop button
            gr.update(interactive=True)  # Re-enable run button
        )


async def run_org_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp

        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        if _global_browser is None:
            _global_browser = Browser(
                config=BrowserConfig(
                    headless=headless,
                    cdp_url=cdp_url,
                    disable_security=disable_security,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None:
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        if _global_agent is None:
            _global_agent = Agent(
                task=task,
                llm=llm,
                use_vision=use_vision,
                browser=_global_browser,
                browser_context=_global_browser_context,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_custom_agent(
        llm,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    try:
        global _global_browser, _global_browser_context, _global_agent

        extra_chromium_args = [f"--window-size={window_w},{window_h}"]
        cdp_url = chrome_cdp
        if use_own_browser:
            cdp_url = os.getenv("CHROME_CDP", chrome_cdp)

            chrome_path = os.getenv("CHROME_PATH", None)
            if chrome_path == "":
                chrome_path = None
            chrome_user_data = os.getenv("CHROME_USER_DATA", None)
            if chrome_user_data:
                extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]
        else:
            chrome_path = None

        controller = CustomController()

        # Initialize global browser if needed
        # if chrome_cdp not empty string nor None
        if (_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url != None):
            _global_browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    cdp_url=cdp_url,
                    chrome_instance_path=chrome_path,
                    extra_chromium_args=extra_chromium_args,
                )
            )

        if _global_browser_context is None or (chrome_cdp and cdp_url != "" and cdp_url != None):
            _global_browser_context = await _global_browser.new_context(
                config=BrowserContextConfig(
                    trace_path=save_trace_path if save_trace_path else None,
                    save_recording_path=save_recording_path if save_recording_path else None,
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=window_w, height=window_h
                    ),
                )
            )

        # Create and run agent
        if _global_agent is None:
            _global_agent = CustomAgent(
                task=task,
                add_infos=add_infos,
                use_vision=use_vision,
                llm=llm,
                browser=_global_browser,
                browser_context=_global_browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=max_actions_per_step,
                tool_calling_method=tool_calling_method,
                max_input_tokens=max_input_tokens,
                generate_gif=True
            )
        history = await _global_agent.run(max_steps=max_steps)

        history_file = os.path.join(save_agent_history_path, f"{_global_agent.state.agent_id}.json")
        _global_agent.save_history(history_file)

        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()

        trace_file = get_latest_files(save_trace_path)

        return final_result, errors, model_actions, model_thoughts, trace_file.get('.zip'), history_file
    except Exception as e:
        import traceback
        traceback.print_exc()
        errors = str(e) + "\n" + traceback.format_exc()
        return '', errors, '', '', None, None
    finally:
        _global_agent = None
        # Handle cleanup based on persistence configuration
        if not keep_browser_open:
            if _global_browser_context:
                await _global_browser_context.close()
                _global_browser_context = None

            if _global_browser:
                await _global_browser.close()
                _global_browser = None


async def run_with_stream(
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens
):
    global _global_agent

    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)
    if not headless:
        result = await run_browser_agent(
            agent_type=agent_type,
            llm_provider=llm_provider,
            llm_model_name=llm_model_name,
            llm_num_ctx=llm_num_ctx,
            llm_temperature=llm_temperature,
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            use_own_browser=use_own_browser,
            keep_browser_open=keep_browser_open,
            headless=headless,
            disable_security=disable_security,
            window_w=window_w,
            window_h=window_h,
            save_recording_path=save_recording_path,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            enable_recording=enable_recording,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            chrome_cdp=chrome_cdp,
            max_input_tokens=max_input_tokens
        )
        # Add HTML content at the start of the result array
        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
        yield [html_content] + list(result)
    else:
        try:
            # Run the browser agent in the background
            agent_task = asyncio.create_task(
                run_browser_agent(
                    agent_type=agent_type,
                    llm_provider=llm_provider,
                    llm_model_name=llm_model_name,
                    llm_num_ctx=llm_num_ctx,
                    llm_temperature=llm_temperature,
                    llm_base_url=llm_base_url,
                    llm_api_key=llm_api_key,
                    use_own_browser=use_own_browser,
                    keep_browser_open=keep_browser_open,
                    headless=headless,
                    disable_security=disable_security,
                    window_w=window_w,
                    window_h=window_h,
                    save_recording_path=save_recording_path,
                    save_agent_history_path=save_agent_history_path,
                    save_trace_path=save_trace_path,
                    enable_recording=enable_recording,
                    task=task,
                    add_infos=add_infos,
                    max_steps=max_steps,
                    use_vision=use_vision,
                    max_actions_per_step=max_actions_per_step,
                    tool_calling_method=tool_calling_method,
                    chrome_cdp=chrome_cdp,
                    max_input_tokens=max_input_tokens
                )
            )

            # Initialize values for streaming
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            final_result = errors = model_actions = model_thoughts = ""
            recording_gif = trace = history_file = None

            # Periodically update the stream while the agent task is running
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(_global_browser_context)
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh ; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception as e:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                if _global_agent and _global_agent.state.stopped:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        recording_gif,
                        trace,
                        history_file,
                        gr.update(value="Stopping...", interactive=False),  # stop_button
                        gr.update(interactive=False),  # run_button
                    ]
                    break
                else:
                    yield [
                        html_content,
                        final_result,
                        errors,
                        model_actions,
                        model_thoughts,
                        recording_gif,
                        trace,
                        history_file,
                        gr.update(),  # Re-enable stop button
                        gr.update()  # Re-enable run button
                    ]
                await asyncio.sleep(0.1)

            # Once the agent task completes, get the results
            try:
                result = await agent_task
                final_result, errors, model_actions, model_thoughts, recording_gif, trace, history_file, stop_button, run_button = result
            except gr.Error:
                final_result = ""
                model_actions = ""
                model_thoughts = ""
                recording_gif = trace = history_file = None

            except Exception as e:
                errors = f"Agent error: {str(e)}"

            yield [
                html_content,
                final_result,
                errors,
                model_actions,
                model_thoughts,
                recording_gif,
                trace,
                history_file,
                stop_button,
                run_button
            ]

        except Exception as e:
            import traceback
            yield [
                f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>",
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),  # Re-enable stop button
                gr.update(interactive=True)  # Re-enable run button
            ]


async def run_deep_search(research_task, max_search_iterations, max_queries_per_iteration, llm_provider, llm_model_name,
                          llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision, use_own_browser,
                          headless, chrome_cdp):
    global _global_agent_state

    # Reset the stop flag
    _global_agent_state.reset_stop_flag()

    # Placeholder for the actual implementation
    import time
    import random
    import markdown

    # Create a markdown file to save the report
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    file_path = f"research_report_{timestamp}.md"

    # Generate a simple markdown report
    markdown_content = f"""# Research Report: {research_task}

## Introduction

This is a research report on the topic: "{research_task}".

## Findings

The research process involved {max_search_iterations} iterations with up to {max_queries_per_iteration} queries per iteration.

### Key Points

1. First key finding about the topic
2. Second key finding about the topic
3. Third key finding about the topic

## Conclusion

Based on the research conducted, we can conclude that this topic requires further investigation.

## References

1. Reference 1
2. Reference 2
3. Reference 3
"""

    # Save the markdown content to a file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)

    # Convert markdown to HTML for display
    html_content = markdown.markdown(markdown_content)

    return markdown_content, file_path, gr.update(value="Stop", interactive=True), gr.update(interactive=True)


def close_global_browser():
    """Close the global browser instance if it exists"""
    global _global_browser, _global_browser_context, _global_agent
    # This is just a placeholder for the actual implementation
    # The actual closing happens in the async context
    return None


def update_llm_num_ctx_visibility(llm_provider):
    """Update the visibility of the context length slider based on the LLM provider"""
    return gr.update(visible=llm_provider == "ollama")


# Map theme names to theme objects
theme_map = {
    "Default": Default(),
    "Citrus": Citrus(),
    "Glass": Glass(),
    "Monochrome": Monochrome(),
    "Ocean": Ocean(),
    "Origin": Origin(),
    "Soft": Soft(),
}


def create_ui(config, theme_name="Ocean", language="en"):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    # Initialize translation - pouze čeština
    import os
    locales_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")
    with Translate(
        languages=["cs"],  # Pouze čeština
        default_language="cs",  # Výchozí jazyk je čeština
        locales_dir=locales_dir  # Path to translations directory
    ) as i18n:
        with gr.Blocks(
                title="Browser Use WebUI", theme=theme_map[theme_name], css=css
        ) as demo:

            with gr.Row():
                gr.Markdown(
                    f"""
                    # 🌐 {i18n("interface.title")}
                    ### {i18n("interface.subtitle")}
                    """,
                    elem_classes=["header-text"],
                )

            with gr.Tabs() as tabs:
                with gr.TabItem(i18n("tabs.agent_settings"), id=1):
                    with gr.Group():
                        agent_type = gr.Radio(
                            ["standardní", "vlastní"],
                            label="Typ agenta",
                            value="standardní" if config['agent_type'] == "org" else "vlastní",
                            info="Vyberte typ agenta, který chcete použít",
                        )
                        with gr.Column():
                            max_steps = gr.Slider(
                                minimum=1,
                                maximum=200,
                                value=config['max_steps'],
                                step=1,
                                label=i18n("agent_settings.max_steps"),
                                info=i18n("agent_settings.max_steps_info"),
                            )
                            max_actions_per_step = gr.Slider(
                                minimum=1,
                                maximum=20,
                                value=config['max_actions_per_step'],
                                step=1,
                                label=i18n("agent_settings.max_actions_per_step"),
                                info=i18n("agent_settings.max_actions_per_step_info"),
                            )
                        with gr.Column():
                            use_vision = gr.Checkbox(
                                label=i18n("agent_settings.use_vision"),
                                value=config['use_vision'],
                                info=i18n("agent_settings.use_vision_info"),
                            )
                            max_input_tokens = gr.Number(
                                label=i18n("agent_settings.max_input_tokens"),
                                value=128000,
                                precision=0

                            )
                            tool_calling_method = gr.Dropdown(
                                label=i18n("agent_settings.tool_calling_method"),
                                value=config['tool_calling_method'],
                                interactive=True,
                                allow_custom_value=True,  # Allow users to input custom model names
                                choices=["auto", "json_schema", "function_calling"],
                                info=i18n("agent_settings.tool_calling_method_info"),
                                visible=False
                            )

                with gr.TabItem(i18n("tabs.llm_settings"), id=2):
                    with gr.Group():
                        llm_provider = gr.Dropdown(
                            choices=[provider for provider, model in utils.model_names.items()],
                            label=i18n("llm_settings.llm_provider"),
                            value=config['llm_provider'],
                            info=i18n("llm_settings.llm_provider_info")
                        )
                        llm_model_name = gr.Dropdown(
                            label=i18n("llm_settings.llm_model"),
                            choices=utils.model_names['openai'],
                            value=config['llm_model_name'],
                            interactive=True,
                            allow_custom_value=True,  # Allow users to input custom model names
                            info=i18n("llm_settings.llm_model_info")
                        )
                        llm_num_ctx = gr.Slider(
                            minimum=2 ** 8,
                            maximum=2 ** 16,
                            value=config['llm_num_ctx'],
                            step=1,
                            label=i18n("llm_settings.llm_num_ctx"),
                            info=i18n("llm_settings.llm_num_ctx_info"),
                            visible=config['llm_provider'] == "ollama"
                        )
                        llm_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=config['llm_temperature'],
                            step=0.1,
                            label=i18n("llm_settings.llm_temperature"),
                            info=i18n("llm_settings.llm_temperature_info")
                        )
                        with gr.Row():
                            llm_base_url = gr.Textbox(
                                label=i18n("llm_settings.llm_base_url"),
                                value=config['llm_base_url'],
                                info=i18n("llm_settings.llm_base_url_info")
                            )
                            llm_api_key = gr.Textbox(
                                label=i18n("llm_settings.llm_api_key"),
                                type="password",
                                value=config['llm_api_key'],
                                info=i18n("llm_settings.llm_api_key_info")
                            )

                with gr.TabItem(i18n("tabs.browser_settings"), id=3):
                    with gr.Group():
                        with gr.Row():
                            use_own_browser = gr.Checkbox(
                                label=i18n("browser_settings.use_own_browser"),
                                value=config['use_own_browser'],
                                info=i18n("browser_settings.use_own_browser_info"),
                            )
                            keep_browser_open = gr.Checkbox(
                                label=i18n("browser_settings.keep_browser_open"),
                                value=config['keep_browser_open'],
                                info=i18n("browser_settings.keep_browser_open_info"),
                            )
                            headless = gr.Checkbox(
                                label=i18n("browser_settings.headless"),
                                value=config['headless'],
                                info=i18n("browser_settings.headless_info"),
                            )
                            disable_security = gr.Checkbox(
                                label=i18n("browser_settings.disable_security"),
                                value=config['disable_security'],
                                info=i18n("browser_settings.disable_security_info"),
                            )
                            enable_recording = gr.Checkbox(
                                label="Povolit nahrávání",
                                value=config['enable_recording'],
                                info="Povolit ukládání nahrávek prohlížeče",
                            )

                        with gr.Row():
                            window_w = gr.Number(
                                label=i18n("browser_settings.window_width"),
                                value=config['window_w'],
                                info="Browser window width",
                            )
                            window_h = gr.Number(
                                label=i18n("browser_settings.window_height"),
                                value=config['window_h'],
                                info="Browser window height",
                            )

                        chrome_cdp = gr.Textbox(
                            label=i18n("browser_settings.chrome_cdp"),
                            placeholder="http://localhost:9222",
                            value="",
                            info=i18n("browser_settings.chrome_cdp_info"),
                            interactive=True,  # Allow editing only if recording is enabled
                        )

                        save_recording_path = gr.Textbox(
                            label="Cesta k nahrávkám",
                            placeholder="např. ./tmp/record_videos",
                            value=config['save_recording_path'],
                            info="Cesta pro ukládání nahrávek prohlížeče",
                            interactive=True,  # Allow editing only if recording is enabled
                        )

                        save_trace_path = gr.Textbox(
                            label="Cesta k trasování",
                            placeholder="např. ./tmp/traces",
                            value=config['save_trace_path'],
                            info="Cesta pro ukládání trasování agenta",
                            interactive=True,
                        )

                        save_agent_history_path = gr.Textbox(
                            label="Cesta k historii agenta",
                            placeholder="např. ./tmp/agent_history",
                            value=config['save_agent_history_path'],
                            info="Určete adresář, kam se bude ukládat historie agenta.",
                            interactive=True,
                        )
                        
                        # Add event handlers for browser settings
                        enable_recording.change(
                            lambda enabled: gr.update(interactive=enabled),
                            inputs=enable_recording,
                            outputs=save_recording_path
                        )
                        
                        use_own_browser.change(fn=close_global_browser)
                        keep_browser_open.change(fn=close_global_browser)

                with gr.TabItem(i18n("tabs.task"), id=4):
                    task = gr.Textbox(
                        label=i18n("task.task_input"),
                        lines=4,
                        placeholder="Zadejte svůj úkol zde...",
                        value="Jdi na google.com a napiš 'OpenAI', klikni na vyhledávání a dej mi první URL",
                        info=i18n("task.task_input_info"),
                    )
                    add_infos = gr.Textbox(
                        label=i18n("task.add_infos"),
                        lines=3,
                        placeholder="Přidejte jakýkoli užitečný kontext nebo instrukce...",
                        info=i18n("task.add_infos_info"),
                    )

                    with gr.Row():
                        run_button = gr.Button(i18n("task.run_button"), variant="primary", scale=2)
                        stop_button = gr.Button(i18n("task.stop_button"), variant="stop", scale=1)

                    with gr.Row():
                        browser_view = gr.HTML(
                            value="<h1 style='width:80vw; height:50vh'>Čekání na spuštění prohlížeče...</h1>",
                            label=i18n("output.browser_view"),
                        )

                    gr.Markdown("### Výsledky")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label=i18n("output.final_result"), lines=3, show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label=i18n("output.errors"), lines=3, show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label=i18n("output.model_actions"), lines=3, show_label=True, visible=False
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label=i18n("output.model_thoughts"), lines=3, show_label=True, visible=False
                            )
                    recording_gif = gr.Image(label="Result GIF", format="gif")
                    trace_file = gr.File(label="Trace File")
                    agent_history_file = gr.File(label="Agent History")
                    
                    # Bind the stop button click event
                    stop_button.click(
                        fn=stop_agent,
                        inputs=[],
                        outputs=[stop_button, run_button],
                    )

                    # Run button click handler
                    run_button.click(
                        fn=run_with_stream,
                        inputs=[
                            agent_type, llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url,
                            llm_api_key,
                            use_own_browser, keep_browser_open, headless, disable_security, window_w, window_h,
                            save_recording_path, save_agent_history_path, save_trace_path,  # Include the new path
                            enable_recording, task, add_infos, max_steps, use_vision, max_actions_per_step,
                            tool_calling_method, chrome_cdp, max_input_tokens
                        ],
                        outputs=[
                            browser_view,  # Browser view
                            final_result_output,  # Final result
                            errors_output,  # Errors
                            model_actions_output,  # Model actions
                            model_thoughts_output,  # Model thoughts
                            recording_gif,  # Latest recording
                            trace_file,  # Trace file
                            agent_history_file,  # Agent history file
                            stop_button,  # Stop button
                            run_button  # Run button
                        ],
                    )

                with gr.TabItem(i18n("tabs.deep_research"), id=5):
                    research_task_input = gr.Textbox(label=i18n("deep_research.research_task"), lines=5,
                                                     value="Vytvořte zprávu o použití zpětnovazebního učení (Reinforcement Learning) pro trénování velkých jazykových modelů, včetně jeho počátků, současného vývoje a budoucích možností, s příklady relevantních modelů a technik. Zpráva by měla obsahovat vlastní postřehy a analýzu, ne jen shrnutí existující literatury.")
                    with gr.Row():
                        max_search_iteration_input = gr.Number(label=i18n("deep_research.max_search_iterations"), value=3,
                                                               precision=0)
                        max_query_per_iter_input = gr.Number(label=i18n("deep_research.max_queries_per_iteration"), value=1,
                                                             precision=0)
                    with gr.Row():
                        research_button = gr.Button(i18n("deep_research.run_research"), variant="primary", scale=2)
                        stop_research_button = gr.Button(i18n("deep_research.stop_research"), variant="stop", scale=1)
                    markdown_output_display = gr.Markdown(label="Výzkumná zpráva")
                    markdown_download = gr.File(label="Stáhnout výzkumnou zprávu")

                    # Bind the stop button click event for research
                    stop_research_button.click(
                        fn=stop_research_agent,
                        inputs=[],
                        outputs=[stop_research_button, research_button],
                    )

                    # Run Deep Research
                    research_button.click(
                        fn=run_deep_search,
                        inputs=[research_task_input, max_search_iteration_input, max_query_per_iter_input, llm_provider,
                                llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key, use_vision,
                                use_own_browser, headless, chrome_cdp],
                        outputs=[markdown_output_display, markdown_download, stop_research_button, research_button]
                    )

            with gr.TabItem("🎥 Recordings", id=7, visible=True):
                def list_recordings(save_recording_path):
                    if not os.path.exists(save_recording_path):
                        return []

                    # Get all video files
                    recordings = glob.glob(os.path.join(save_recording_path, "*.[mM][pP]4")) + glob.glob(
                        os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

                    # Sort recordings by creation time (oldest first)
                    recordings.sort(key=os.path.getctime)

                    # Add numbering to the recordings
                    numbered_recordings = []
                    for idx, recording in enumerate(recordings, start=1):
                        filename = os.path.basename(recording)
                        numbered_recordings.append((recording, f"{idx}. {filename}"))

                    return numbered_recordings

                recordings_gallery = gr.Gallery(
                    label=i18n("recordings.recordings_gallery"),
                    value=list_recordings(config['save_recording_path']),
                    columns=3,
                    height="auto",
                    object_fit="contain"
                )

                refresh_button = gr.Button(i18n("recordings.refresh_recordings"), variant="secondary")
                refresh_button.click(
                    fn=lambda: list_recordings(config['save_recording_path']),
                    inputs=[],
                    outputs=recordings_gallery
                )

            with gr.TabItem(i18n("tabs.ui_configuration"), id=8):
                config_file_input = gr.File(
                    label=i18n("ui_configuration.load_config_file"),
                    file_types=[".pkl"],
                    interactive=True
                )
                with gr.Row():
                    load_config_button = gr.Button(i18n("ui_configuration.load_config_button"), variant="primary")
                    save_config_button = gr.Button(i18n("ui_configuration.save_config_button"), variant="primary")

                config_status = gr.Textbox(
                    label=i18n("ui_configuration.status"),
                    lines=2,
                    interactive=False
                )

                # Attach the callback to the LLM provider dropdown
                llm_provider.change(
                    lambda provider, api_key, base_url: update_model_dropdown(provider, api_key, base_url),
                    inputs=[llm_provider, llm_api_key, llm_base_url],
                    outputs=llm_model_name
                )

                # Bind the change event of llm_provider to update the visibility of context length slider
                llm_provider.change(
                    fn=update_llm_num_ctx_visibility,
                    inputs=llm_provider,
                    outputs=llm_num_ctx
                )

            # Bind the config buttons at the end of the function
            load_config_button.click(
                fn=update_ui_from_config,
                inputs=[config_file_input],
                outputs=[
                    agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                    llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                    use_own_browser, keep_browser_open, headless, disable_security, enable_recording,
                    window_w, window_h, save_recording_path, save_trace_path, save_agent_history_path,
                    task, config_status
                ]
            )

            save_config_button.click(
                fn=save_current_config,
                inputs=[
                    agent_type, max_steps, max_actions_per_step, use_vision, tool_calling_method,
                    llm_provider, llm_model_name, llm_num_ctx, llm_temperature, llm_base_url, llm_api_key,
                    use_own_browser, keep_browser_open, headless, disable_security,
                    enable_recording, window_w, window_h, save_recording_path, save_trace_path,
                    save_agent_history_path, task,
                ],
                outputs=[config_status]
            )

    return demo


def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    # Parametr --language je ignorován, vždy používáme češtinu
    parser.add_argument("--language", type=str, default="cs", help="Ignorováno - vždy používáme češtinu")
    args = parser.parse_args()

    config_dict = default_config()

    # Vždy používáme češtinu
    demo = create_ui(config_dict, theme_name=args.theme, language="cs")
    demo.launch(server_name=args.ip, server_port=args.port)


if __name__ == '__main__':
    main()