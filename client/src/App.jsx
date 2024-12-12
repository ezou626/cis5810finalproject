import React, { useState } from "react";
import CaptionReader from "./components/CaptionReader";
import DownloadButton from "./components/DownloadButton";
import LandingPage from "./components/LandingPage";
import VideoStreamer from "./components/VideoStreamer";

export default function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState("");
  const [firstPage, setFirstPage] = useState(true);

  const streamUrl = isStreaming ? videoUrl : "";

  const startStreaming = () => {
    if (!videoUrl.trim()) {
      setError("Please provide a valid video URL.");
      return;
    }
    setError("");
    setIsStreaming(true);
  };

  const stopStreaming = () => {
    setIsStreaming(false);
  };

  return (
    firstPage ? 
    <LandingPage onGetStarted={() => setFirstPage(false)} /> 
    :
    (
    <main className="bg-gray-900 text-white min-h-screen">
      {/* Header Section */}
      <header className="py-16 text-center bg-gray-800">
        <h1 className="text-5xl md:text-6xl font-extrabold leading-tight">
          Stephen AI Smith
        </h1>
      </header>

      <section className="py-12">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-center mb-12">Start Your Experience</h2>

          <div className="flex justify-center">
            <div className="w-full max-w-lg">
              <label
                htmlFor="videoUrl"
                className="block text-lg font-medium text-gray-200 mb-2"
              >
                Enter an MP4 Video URL
              </label>
              <input
                id="videoUrl"
                type="text"
                value={videoUrl}
                onChange={(e) => setVideoUrl(e.target.value)}
                className="w-full px-4 py-3 rounded-lg border border-gray-500 bg-gray-800 text-white shadow-sm focus:border-indigo-500 focus:ring-indigo-500 placeholder-gray-500"
                placeholder="https://example.com/video.mp4"
              />
              {error && <p className="text-red-500 text-sm mt-2">{error}</p>}
            </div>
          </div>

          <div className="mt-8 flex justify-center space-x-4">
            <button
              onClick={startStreaming}
              className={`px-8 py-4 text-lg font-bold rounded-md shadow-md transition ${
                streamUrl 
                  ? "bg-gray-500 cursor-not-allowed text-gray-300" 
                  : "bg-indigo-600 hover:bg-indigo-700 text-white"
              }`}
              disabled={!!streamUrl}
            >
              Start Streaming
            </button>
            <button
              onClick={stopStreaming}
              className={`px-8 py-4 text-lg font-bold rounded-md shadow-md transition ${
                !streamUrl 
                  ? "bg-gray-500 cursor-not-allowed text-gray-300" 
                  : "bg-red-600 hover:bg-red-700 text-white"
              }`}
              disabled={!streamUrl}
            >
              Stop Streaming
            </button>
          </div>
        </div>
      </section>

      <section className="py-24 bg-gray-800">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-4xl font-bold text-center mb-12">Your Video Experience</h2>

          <div className="flex flex-col md:flex-row gap-12">
            <VideoStreamer
              streamUrl={streamUrl}
              startStreaming={startStreaming}
              stopStreaming={stopStreaming}
            />
            <CaptionReader streamUrl={streamUrl} />
          </div>

          <div className="mt-12 flex justify-center">
            <DownloadButton
              downloadUrl="http://localhost:8000/download_processed_video"
              filename="processed_video.mp4"
            />
          </div>
        </div>
      </section>

      {/* Footer Section */}
      <footer className="bg-gray-900 py-6">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <p className="text-gray-400">
            &copy; 2024 ZoomBot AI. All Rights Reserved.
          </p>
        </div>
      </footer>
    </main>
    )
  );
}
