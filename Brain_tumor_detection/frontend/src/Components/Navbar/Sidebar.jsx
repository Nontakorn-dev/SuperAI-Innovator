import { Link, useLocation } from "react-router-dom";
import React, { useRef } from "react";
import { useNavigate } from "react-router-dom";
import ButtonStyles from "./navbarButton.module.css"
import ExportButton from "./ExportButton";

export default function Sidebar({ chats, setChats, setShowModal }) {
  const location = useLocation();
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  const handleUploadClick = () => {
    fileInputRef.current.click();
  };

  async function deleteFile(fileUrl) {
    try {
      // Convert URL to relative path (e.g., remove http://localhost:8000/)
      const relativePath = fileUrl.replace("http://localhost:8000/", "");
  
      const res = await fetch(`http://localhost:8000/delete_file?filepath=static/${relativePath}`, {
        method: "DELETE",
      });
  
      const result = await res.json();
  
      if (res.ok) {
        console.log("✅ File deleted:", result.detail);
      } else {
        console.error("❌ Failed to delete:", result.detail);
      }
    } catch (error) {
      console.error("❌ Error deleting file:", error.message);
    }
  }

  async function removeChat(id) {
    // 1. Find the chat with the given id
    const chat = chats.find(c => c.id === id);
  
    if (!chat) {
      console.warn(`Chat with id ${id} not found.`);
      return;
    }
  
    // 2. Delete all related image files
    if (chat.content?.viewerImages?.length) {
      for (const img of chat.content.viewerImages) {
        await deleteFile(img); // assuming img.image is the URL
      }
    }
  
    // 3. Remove the chat from state
    setChats(prev => prev.filter(c => c.id !== id));
  }  

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (!file) return;

    if (!file.name.endsWith(".json")) {
      alert("Please upload a JSON file.");
      return;
    }

    const reader = new FileReader();

    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target.result);

        if (!json.id || !json.topic) {
          alert("Invalid chat data.");
          return;
        }

        setChats((prev) => [...prev, json]);
        navigate(`/chat/${json.id}`);

      } catch (err) {
        alert("Failed to parse JSON file.");
      }
    };

    reader.readAsText(file);
    event.target.value = null;
  };

  return (
    <div className={ButtonStyles["sidebar-container"]}>
      {/* Header with logo */}
      <div className={ButtonStyles["sidebar-header"]}>
        <Link to="/" className={ButtonStyles["logo-link"]}>
          <div className={ButtonStyles["logo-container"]}>
            <img src="/brain_icon.png" alt="brain" width="60px" height="40px" />
            {/* <span className={ButtonStyles["logo-text"]}>Brain AI</span> */}
          </div>
        </Link>
      </div>

      {/* Action buttons */}
      <div className={ButtonStyles["action-buttons"]}>
        <button 
          className={ButtonStyles["new-case-button"]} 
          onClick={() => setShowModal(true)}
        >
          <span className={ButtonStyles["button-icon"]}>+</span>
          New Case
        </button>
        
        <button 
          className={ButtonStyles["upload-button"]} 
          onClick={handleUploadClick}
        >
          <span className={ButtonStyles["button-icon"]}>📁</span>
          Upload Chat
        </button>
        
        <input
          type="file"
          accept=".json"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={{ display: "none" }}
        />
      </div>

      {/* Cases section */}
      <div className={ButtonStyles["cases-section"]}>
        <h3 className={ButtonStyles["section-title"]}>Recent Cases</h3>
        
        <div className={ButtonStyles["cases-list"]}>
          {chats.map((chat, index) => {
            const isActive = location.pathname === `/chat/${chat.id}`;
            return (
              <div
                key={index}
                className={`${ButtonStyles["chat-item"]} ${isActive ? ButtonStyles["active-chat"] : ""}`}
              >
                <Link className={ButtonStyles["chat-link"]} to={`/chat/${chat.id}`}>
                  <div className={ButtonStyles["chat-content"]}>
                    <ExportButton chat={chat} />
                    
                    <div className={ButtonStyles["chat-info"]}>
                      <div className={ButtonStyles["chat-topic"]} title={chat.topic}>
                        {chat.topic}
                      </div>
                      <div className={ButtonStyles["chat-details"]}>
                        {chat.content.selectedDimension} • Case
                      </div>
                    </div>
                    
                    <button 
                      className={ButtonStyles["close-button"]} 
                      onClick={(e) => {
                        e.preventDefault();
                        e.stopPropagation();
                        removeChat(chat.id);
                      }}
                      title="Remove case"
                    >
                      ×
                    </button>
                  </div>
                </Link>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}