import React from "react";

const VideoStreamer = ({ streamUrl }) => {
  return (
    <div className="flex flex-col items-center w-full md:w-2/3 bg-white rounded-lg shadow-lg p-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Video Stream</h2>

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
