import React from "react";

const DownloadButton = ({ downloadUrl, filename }) => {
  return (
    <a href={downloadUrl} download={filename}>
      <button className="px-4 py-2 bg-blue-500 text-white rounded-md">
        Download Processed Video
      </button>
    </a>
  );
};

export default DownloadButton;
