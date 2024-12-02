import React, { useState } from "react";

const backendUrl = import.meta.env.VITE_BACKEND_URL;

const VideoStreamer = () => {
  const [videoUrl, setVideoUrl] = useState("");
  const [streamUrl, setStreamUrl] = useState("");
  const [error, setError] = useState("");

  const startStreaming = () => {
    if (!videoUrl.trim()) {
      setError("Please provide a valid video URL.");
      return;
    }
    setError("");
    setStreamUrl(`${backendUrl}/video?url=${encodeURIComponent(videoUrl)}`);
  };

  const stopStreaming = () => {
    setStreamUrl("");
  };

  return (
    <div className="flex flex-col items-center p-4 space-y-6">

      <div className="w-full max-w-md">
        <label className="block text-sm font-medium text-gray-700">
          Video URL
        </label>
        <input
          type="text"
          value={videoUrl}
          onChange={(e) => setVideoUrl(e.target.value)}
          className="mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-indigo-500 focus:ring-indigo-500 sm:text-sm"
          placeholder="Enter an MP4 video URL"
        />
      </div>

      <div className="flex space-x-4">
        <button
          onClick={startStreaming}
          className={`px-4 py-2 rounded-md text-white ${
            streamUrl
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-indigo-500 hover:bg-indigo-600"
          }`}
          disabled={!!streamUrl}
        >
          Start Streaming
        </button>
        <button
          onClick={stopStreaming}
          className={`px-4 py-2 rounded-md text-white ${
            !streamUrl
              ? "bg-gray-400 cursor-not-allowed"
              : "bg-red-500 hover:bg-red-600"
          }`}
          disabled={!streamUrl}
        >
          Stop Streaming
        </button>
      </div>

      {error && <p className="text-red-500 text-sm">{error}</p>}

      <div className="w-full max-w-3xl mt-8">
        {streamUrl ? (
          <video
            src={streamUrl}
            controls
            autoPlay
            className="w-full aspect-video bg-black"
          />
        ) : (
          <p className="text-gray-500">No live stream is currently active.</p>
        )}
      </div>
    </div>
  );
};

export default VideoStreamer;