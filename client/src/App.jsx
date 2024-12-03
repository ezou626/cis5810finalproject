import React, { useState } from "react";
import CaptionReader from "./components/CaptionReader";
import VideoStreamer from "./components/VideoStreamer";

export default function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [isStreaming, setIsStreaming] = useState("");
  const [error, setError] = useState("");

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
    <main className="min-h-screen bg-gray-50 flex flex-col items-center py-10 space-y-8">
      <h1 className="text-3xl font-bold text-indigo-600">Auto-Commentator</h1>

      <div className="w-full max-w-lg">
        <label
          htmlFor="videoUrl"
          className="block text-lg font-medium text-gray-700 mb-2"
        >
          Enter an MP4 Video URL
        </label>
        <input
          id="videoUrl"
          type="text"
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
          className="w-full px-4 py-2 rounded-lg border border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 text-gray-700"
          placeholder="https://example.com/video.mp4"
        />
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      <div className="flex space-x-4 mb-4">
        <button
          onClick={startStreaming}
          className={`px-6 py-2 rounded-lg text-white font-semibold transition ${
            
              streamUrl ? "bg-gray-400 cursor-not-allowed"
              : "bg-indigo-500 hover:bg-indigo-600"
          }`}
          disabled={!!streamUrl}
        >
          Start Streaming
        </button>
        <button
          onClick={stopStreaming}
          className={`px-6 py-2 rounded-lg text-white font-semibold transition ${
            !streamUrl
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-red-500 hover:bg-red-600"
          }`}
          disabled={!streamUrl}
        >
          Stop Streaming
        </button>
      </div>

      <div className="w-full max-w-5xl flex flex-col md:flex-row gap-8">
        <VideoStreamer
          streamUrl={streamUrl}
          startStreaming={startStreaming}
          stopStreaming={stopStreaming}
        />
        <CaptionReader streamUrl={streamUrl} />
        
      </div>
    </main>
  );
}
