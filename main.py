"""Streamlit webapp module"""
from os.path import join
from os import listdir, remove
import gc

import streamlit as st
import torch
from pesq import pesq
import numpy as np
from scipy import signal
import soundfile as sf
import resampy
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data.model.dccrn_model import DCCRN


FS = 16000
NO_NOISE_STR = "None"
MAX_AUDIO_LENGTH = 9 * FS


def main():
    """Streamlit webapp flow"""
    ## setting up main layout
    st.title("Speech Enhancement")
    st.write(
        "This app lets you play around with a DNN speech enhancement model. You can upload your own (noisy) speech file and optionally add (more) noise. Note that the signal to noise ratio (SNR) slider assumes clean speech input."
    )
    progress_slot = st.empty()
    plotting_slot = st.empty()

    ## setting up sidebar
    st.sidebar.header("Obtain speech")
    uploaded_file = st.sidebar.file_uploader(
        "Use the example, or upload your own file", type="wav"
    )
    audio_trimmed_warning_slot = st.sidebar.empty()
    reverberant = st.sidebar.checkbox("Make reverberant")
    st.sidebar.header("Add noise")
    snr_slot = st.sidebar.empty()
    noise_file_name_slot = st.sidebar.empty()
    st.sidebar.header("Other settings")
    model_id = st.sidebar.radio("Model", ["A", "B"])
    signal_presentation = st.sidebar.radio("Plot type", ["Time signal", "Spectogram"])

    ## load and prepare input
    progress_slot.info("Status: generating noisy audio...")
    # load speech
    if uploaded_file is not None:
        # load uploaded file
        speech_samples = load_audio(
            uploaded_file, warning_slot=audio_trimmed_warning_slot
        )
        noise_file_name = noise_file_name_slot.radio(
            "Noise type", get_noises(), index=0
        )
    else:
        # load speech example file
        speech_samples = load_audio(join("data", "speech", "random_speech_sample.wav"))
        noise_file_name = noise_file_name_slot.radio(
            "Noise type", get_noises(), index=1
        )
    # make reverberant
    if reverberant:
        rir = load_audio(join("data", "rir", "random_rir.wav"))
        speech_samples = make_reverberant(speech_samples, rir)
    pesq_score_clean = pesq(FS, speech_samples, speech_samples, "wb")

    # add noise
    noise_samples = load_audio(join("data", "noise", noise_file_name + ".wav"))
    if noise_samples is None:
        snr_slot.empty()
        snr = None
    else:
        snr = snr_slot.slider(
            "SNR [dB]", min_value=-5.0, max_value=20.0, value=5.0, step=1.0
        )
    noisy_samples = mix_audio(speech_samples, noise_samples, snr)
    pesq_score_noisy = pesq(FS, noisy_samples, speech_samples, "wb")

    ## show input
    fig = None
    row = 1
    if noise_samples is not None:
        progress_slot.info("Status: plotting original audio...")

        fig = present_audio(
            speech_samples,
            "original",
            fig,
            plotting_slot,
            row,
            signal_presentation,
            pesq_score_clean,
        )
        row += 1
    progress_slot.info("Status: plotting input audio...")
    fig = present_audio(
        noisy_samples,
        "input",
        fig,
        plotting_slot,
        row,
        signal_presentation,
        pesq_score_noisy,
    )
    row += 1

    ## run model
    progress_slot.info("Status: loading model...")
    model = load_model(model_id)
    progress_slot.info("Status: running model on noisy audio...")
    cleaned_samples = run_model(model, noisy_samples)
    cleaned_samples = (
        np.max(speech_samples) / np.max(cleaned_samples) * cleaned_samples * -1
    )

    ## show output
    progress_slot.info("Status: plotting enhanced audio...")
    pesq_score_enhanced = pesq(FS, speech_samples, cleaned_samples, "wb")
    fig = present_audio(
        cleaned_samples,
        "enhanced",
        fig,
        plotting_slot,
        row,
        signal_presentation,
        pesq_score_enhanced,
    )
    progress_slot.info("Status: done!")

    ## collect garbage
    gc.collect()

    ## add copyright statement
    st.markdown(
        "Copyright &copy; 2021 Femke B. Gelderblom.  [ResearchGate profile](https://www.researchgate.net/profile/Femke_Gelderblom)"
    )


def run_model(model, noisy_samples):
    """Run pretrained model on noisy"""
    torch.set_grad_enabled(False)
    return np.squeeze((model(torch.from_numpy(noisy_samples))).detach().cpu().numpy())


def load_audio(file_path, warning_slot=None):
    """Read audio samples from file"""
    # return if no noise option selected
    if file_path == join("data", "noise", NO_NOISE_STR + ".wav"):
        return None
    # load audio
    audio_samples, fs = sf.read(file_path, dtype="float32", always_2d=True)
    audio_samples = audio_samples[:, 0]
    # issue warning and clip aduio if audio too long
    if len(audio_samples) > MAX_AUDIO_LENGTH // FS * fs:
        audio_samples = audio_samples[0 : MAX_AUDIO_LENGTH // FS * fs]
        if warning_slot is not None:
            warning_slot.warning(
                "Long audio segment detected: automatically trimmed to "
                + str(MAX_AUDIO_LENGTH // FS)
                + " seconds."
            )
    # resample
    if fs != FS:
        audio_samples = resampy.resample(audio_samples, fs, FS)

    return np.squeeze(audio_samples)


def mix_audio(speech_samples, noise_samples, desired_snr):
    "Mix audio to obtain noisy audio with desired snr"
    # no need to mix if there is no noise
    if noise_samples is None:
        return speech_samples
    # cut audio signal to length of the shortest signal
    if len(speech_samples) < len(noise_samples):
        noise_samples = noise_samples[0 : len(speech_samples)]
    else:
        speech_samples = speech_samples[0 : len(noise_samples)]
    # calculate scaling factor to mix at desired snr
    speech_rms = (speech_samples ** 2).mean() ** 0.5
    noise_rms = (noise_samples ** 2).mean() ** 0.5
    scaling_factor = (
        speech_rms / (10 ** (desired_snr / 20)) / (noise_rms + np.finfo(float).eps)
    )
    # mix and return
    return speech_samples + np.multiply(noise_samples, scaling_factor)


def make_reverberant(speech_samples, rir):
    """Make speech reverberant"""
    reverb_speech = signal.fftconvolve(speech_samples, rir * 0.5, mode="full")

    # make reverb_speech same length as clean_speech
    reverb_speech = reverb_speech[0 : speech_samples.shape[0]]

    return reverb_speech


def get_noises():
    "Get list of available noise types"
    # Obtain all files in the noise directory
    file_names = listdir(join("data", "noise"))
    # remove extensions
    file_names_without_ext = [file_name.split(".")[0] for file_name in file_names]
    # add no noise option
    file_names_without_ext.insert(0, NO_NOISE_STR)
    return file_names_without_ext


def present_audio(
    audio_samples, label, fig, plotting_slot, row, signal_presentation, pesq_score
):
    "Show audio player and plot"
    # show audio player
    st.header(label)
    st.write("pesq: " + str(round(pesq_score, 2)))
    temp_file_name = "temp.wav"
    sf.write(temp_file_name, audio_samples.T, FS)
    st.audio(temp_file_name)
    remove(temp_file_name)
    # plot audio in subplot
    if fig is None:
        if label == "original":
            n_rows = 3
            titles = ["original", "input", "enhanced"]
            height = 600
        else:
            n_rows = 2
            titles = ["input", "enhanced"]
            height = 400
        fig = make_subplots(
            rows=n_rows,
            cols=1,
            subplot_titles=titles,
            shared_xaxes=True,
        )
        fig.update_layout(showlegend=False, height=height)
        if signal_presentation == "Time signal":
            fig.update_yaxes(fixedrange=True, range=[-1.0, 1.0], title="level")
        else:
            fig.update_yaxes(fixedrange=True, title="Frequency Hz")
        fig.update_xaxes(title="time [s]")
    time = np.arange(len(audio_samples)) / FS
    if signal_presentation == "Time signal":
        fig.append_trace(
            go.Scatter(x=time, y=audio_samples, line=dict(width=1)),
            row=row,
            col=1,
        )
    else:
        freqs, bins, Pxx = signal.spectrogram(
            audio_samples,
            fs=FS,
            window="hann",
            nfft=512,
        )
        fig.append_trace(
            go.Heatmap(
                x=bins, y=freqs, z=10 * np.log10(Pxx), colorscale="Jet", showscale=False
            ),
            row=row,
            col=1,
        )
    plotting_slot.plotly_chart(fig, use_container_width=True)
    return fig


@st.cache
def load_model(model_id):
    """Load pretrained model"""
    return DCCRN.load(model_id)


if __name__ == "__main__":
    main()
