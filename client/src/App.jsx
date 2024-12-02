import React, { useState } from "react";
import CaptionReader from "./components/CaptionReader";
import VideoStreamer from "./components/VideoStreamer";

const backendUrl = import.meta.env.VITE_BACKEND_URL;

export default function App() {
  const [videoUrl, setVideoUrl] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [error, setError] = useState("");

  const startStreaming = () => {
    if (!videoUrl.trim()) {
      setError("Please provide a valid video URL.");
      return;
    }
    setError("");
    setStreamUrl(`${backendUrl}/play_video?url=${encodeURIComponent(videoUrl)}`);
  };

  const stopStreaming = () => {
    setStreamUrl("");
  };

  return (
    <main className="min-h-screen bg-gray-50 flex flex-col items-center py-10 space-y-8">
      <h1 className="text-3xl font-bold text-indigo-600">Live Video Streamer</h1>

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

      <div className="w-full max-w-5xl flex flex-col md:flex-row gap-8">
        <VideoStreamer
          streamUrl={streamUrl}
          startStreaming={startStreaming}
          stopStreaming={stopStreaming}
        />
        <div>
          <h1 className="text-xl font-semibold text-gray-900">Server-Sent Event Stream</h1>
          {streamUrl && <CaptionReader videoUrl={videoUrl} />}
        </div>
        
      </div>
    </main>
  );
}
