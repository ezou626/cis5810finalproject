import React from "react";

const VideoStreamer = ({ streamUrl, imgKey }) => {

  const modifiedStreamUrl = streamUrl
    ? `http://localhost:8000/play_video_mod?url=${encodeURIComponent(streamUrl)}&imgKey=${imgKey}`
    : null;

  return (
    <div className="flex flex-col items-center w-full md:w-2/3 bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Video Stream</h2>

      <div className="w-full">
        {modifiedStreamUrl ? (
          <img
            key={imgKey}
            src={modifiedStreamUrl}
            alt="Processed Stream"
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
