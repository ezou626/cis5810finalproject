import React from "react";

const VideoStreamer = ({ streamUrl, startStreaming, stopStreaming }) => {
  return (
    <div className="flex flex-col items-center w-full md:w-2/3 bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Video Stream</h2>

      <div className="flex space-x-4 mb-4">
        <button
          onClick={startStreaming}
          className={`px-6 py-2 rounded-lg text-white font-semibold transition ${
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

      <div className="w-full">
        {streamUrl ? (
          <video
            src={streamUrl}
            controls
            autoPlay
            className="w-full rounded-lg bg-black shadow-md"
          />
        ) : (
          <p className="text-gray-500">No live stream is currently active.</p>
        )}
      </div>
    </div>
  );
};

export default VideoStreamer;
