import { useState, useRef, DragEvent, ChangeEvent } from "react";
import Link from "next/link";
import {
  MessageSquare,
  FileText,
  UploadCloud,
  File,
  CheckCircle,
  AlertTriangle,
  ArrowLeft,
  Loader2,
  Sparkles
} from "lucide-react";
import { uploadDocument, UploadResponse } from "../lib/api";

interface UploadedItem {
  id: string;
  filename: string;
  markdownFile: string;
  timestamp: Date;
  status: "success" | "error";
  details?: string;
}

export default function Admin() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  
  // Keep track of files processed in current session
  const [history, setHistory] = useState<UploadedItem[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    setError(null);
    setSuccess(null);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
    }
  };

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    setError(null);
    setSuccess(null);
    if (e.target.files && e.target.files.length > 0) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  const handleUpload = async () => {
    if (!selectedFile) return;
    setIsUploading(true);
    setProgress(0);
    setError(null);
    setSuccess(null);

    try {
      const result: UploadResponse = await uploadDocument(selectedFile, (pct) => {
        setProgress(pct);
      });

      const chunks = (result as any).chunks_ingested ?? 0;
      setSuccess(
        `"${result.filename}" processed → "${result.markdown_file}" · ${chunks} chunks embedded into vector store.`
      );

      // Add to session history
      const newItem: UploadedItem = {
        id: Math.random().toString(36).substring(7),
        filename: result.filename,
        markdownFile: result.markdown_file,
        timestamp: new Date(),
        status: "success",
      };
      setHistory((prev) => [newItem, ...prev]);
      setSelectedFile(null);
    } catch (err: any) {
      const errorMessage = err.message || "Failed to process the document.";
      setError(errorMessage);

      // Add failed item to session history
      const newItem: UploadedItem = {
        id: Math.random().toString(36).substring(7),
        filename: selectedFile.name,
        markdownFile: "",
        timestamp: new Date(),
        status: "error",
        details: errorMessage,
      };
      setHistory((prev) => [newItem, ...prev]);
    } finally {
      setIsUploading(false);
    }
  };

  const formatBytes = (bytes: number, decimals = 2) => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  };

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 glass-panel border-r border-slate-800 flex flex-col justify-between shrink-0 hidden md:flex">
        <div className="p-6">
          {/* Logo Header */}
          <div className="flex items-center space-x-2.5 mb-8">
            <div className="p-2 bg-violet-600 rounded-xl shadow-lg shadow-violet-500/30">
              <Sparkles className="h-5 w-5 text-white" />
            </div>
            <div>
              <h1 className="text-sm font-semibold tracking-wide bg-gradient-to-r from-white to-slate-400 bg-clip-text text-transparent">
                M-RAG Assistant
              </h1>
              <p className="text-[10px] text-violet-400 font-medium">Local Workspace</p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="space-y-1.5">
            <Link
              href="/"
              className="flex items-center space-x-3 px-4 py-3 rounded-xl text-slate-400 hover:text-slate-200 hover:bg-slate-900 border border-transparent hover:border-slate-800/50 font-medium text-sm transition-all"
            >
              <MessageSquare className="h-4.5 w-4.5" />
              <span>Chat Assistant</span>
            </Link>
            <Link
              href="/admin"
              className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-violet-600/10 text-violet-400 border border-violet-500/20 font-medium text-sm transition-all"
            >
              <FileText className="h-4.5 w-4.5" />
              <span>Document Upload</span>
            </Link>
          </nav>
        </div>

        <div className="p-6 border-t border-slate-900">
          <Link
            href="/"
            className="flex items-center justify-center space-x-2 w-full py-2.5 border border-slate-850 rounded-xl text-xs bg-slate-900/30 text-slate-400 hover:text-slate-200 hover:bg-slate-900 transition-all font-medium"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            <span>Back to Chat</span>
          </Link>
        </div>
      </aside>

      {/* Main Panel */}
      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        <header className="h-16 border-b border-slate-900 flex items-center justify-between px-6 md:px-8 bg-slate-950/80 backdrop-blur-md shrink-0">
          <h2 className="text-sm font-semibold tracking-wide text-slate-100 flex items-center">
            Document Ingestion Dashboard
          </h2>
        </header>

        <div className="max-w-4xl mx-auto w-full p-6 md:p-8 space-y-6">
          {/* Main content grid */}
          <div className="grid grid-cols-1 gap-6">
            
            {/* Upload Box Card */}
            <div className="glass-card rounded-3xl p-6 md:p-8 space-y-6">
              <div>
                <h3 className="text-md font-semibold text-slate-100">Ingest New Manuals</h3>
                <p className="text-xs text-slate-400 mt-1">
                  Upload raw manuals to convert them to markdown using Microsoft MarkItDown.
                  Supported formats: PDF, DOCX, PPT/PPTX, XLS/XLSX, TXT.
                </p>
              </div>

              {/* Drag and Drop Zone */}
              <div
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={triggerFileSelect}
                className={`border-2 border-dashed rounded-2xl p-8 flex flex-col items-center justify-center cursor-pointer transition-all duration-300 ${
                  isDragging
                    ? "border-violet-500 bg-violet-600/5 shadow-inner"
                    : "border-slate-800 hover:border-slate-700 bg-slate-900/25"
                }`}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  accept=".pdf,.docx,.ppt,.pptx,.xls,.xlsx,.txt"
                  className="hidden"
                />
                <div className="p-4 bg-slate-900 rounded-2xl border border-slate-800 shadow-sm text-slate-400 mb-4">
                  <UploadCloud className="h-6 w-6 text-violet-400" />
                </div>
                <p className="text-sm font-medium text-slate-200 text-center">
                  Drag and drop your document here, or <span className="text-violet-400 hover:text-violet-300 underline">browse</span>
                </p>
                <p className="text-[10px] text-slate-500 mt-1.5">Max size 25MB. PDF, DOCX, PPT/PPTX, XLS/XLSX, TXT</p>
              </div>

              {/* Selected File Details */}
              {selectedFile && (
                <div className="bg-slate-900/50 border border-slate-850 rounded-2xl p-4 flex items-center justify-between animate-fadeIn">
                  <div className="flex items-center space-x-3.5 min-w-0">
                    <div className="p-2.5 bg-slate-800 rounded-xl border border-slate-750 text-violet-400">
                      <File className="h-5 w-5" />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-200 truncate">{selectedFile.name}</p>
                      <p className="text-xs text-slate-500 mt-0.5">{formatBytes(selectedFile.size)}</p>
                    </div>
                  </div>
                  <button
                    onClick={handleUpload}
                    disabled={isUploading}
                    className="flex items-center space-x-1.5 px-4.5 py-2.5 rounded-xl text-xs font-semibold text-white bg-violet-600 hover:bg-violet-500 shadow-md shadow-violet-500/10 active:scale-95 transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
                  >
                    {isUploading ? (
                      <>
                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        <span>Uploading...</span>
                      </>
                    ) : (
                      <span>Start Upload</span>
                    )}
                  </button>
                </div>
              )}

              {/* Progress indicator */}
              {isUploading && (
                <div className="space-y-2.5 animate-fadeIn">
                  <div className="flex justify-between text-xs font-medium text-slate-400">
                    <span>Uploading and Parsing Document...</span>
                    <span>{progress}%</span>
                  </div>
                  <div className="h-2 w-full bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                    <div
                      className="h-full bg-gradient-to-r from-violet-600 to-indigo-500 rounded-full transition-all duration-300"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}

              {/* Status Alert Panels */}
              {success && (
                <div className="p-4 bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 rounded-2xl text-xs flex items-start space-x-3 animate-fadeIn">
                  <CheckCircle className="h-5 w-5 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-slate-200">Processing Success</p>
                    <p className="mt-1 leading-relaxed">{success}</p>
                  </div>
                </div>
              )}

              {error && (
                <div className="p-4 bg-rose-500/10 border border-rose-500/20 text-rose-400 rounded-2xl text-xs flex items-start space-x-3 animate-fadeIn">
                  <AlertTriangle className="h-5 w-5 shrink-0 mt-0.5" />
                  <div>
                    <p className="font-semibold text-slate-200">Processing Failed</p>
                    <p className="mt-1 leading-relaxed">{error}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Upload History List */}
            <div className="glass-card rounded-3xl p-6 md:p-8 space-y-6">
              <div>
                <h3 className="text-md font-semibold text-slate-100">Session Processing Log</h3>
                <p className="text-xs text-slate-400 mt-1">
                  History of files processed during this administrative session.
                </p>
              </div>

              {history.length === 0 ? (
                <div className="text-center py-10 border border-slate-900/50 rounded-2xl bg-slate-900/10">
                  <FileText className="h-7 w-7 text-slate-600 mx-auto mb-2.5" />
                  <p className="text-xs text-slate-500">No documents uploaded in this session yet.</p>
                </div>
              ) : (
                <div className="divide-y divide-slate-900 border border-slate-900 rounded-2xl overflow-hidden bg-slate-900/10">
                  {history.map((item) => (
                    <div
                      key={item.id}
                      className="p-4 flex flex-col sm:flex-row sm:items-center sm:justify-between text-xs hover:bg-slate-900/20 transition-all gap-2"
                    >
                      <div className="flex items-start space-x-3 min-w-0">
                        <div
                          className={`p-2 rounded-lg border shrink-0 ${
                            item.status === "success"
                              ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
                              : "bg-rose-500/10 border-rose-500/20 text-rose-400"
                          }`}
                        >
                          <File className="h-4 w-4" />
                        </div>
                        <div className="min-w-0">
                          <p className="font-medium text-slate-200 truncate">{item.filename}</p>
                          <p className="text-[10px] text-slate-500 mt-0.5">
                            {item.timestamp.toLocaleTimeString()} -{" "}
                            {item.status === "success"
                              ? `Processed -> ${item.markdownFile}`
                              : `Failed`}
                          </p>
                        </div>
                      </div>
                      <div className="shrink-0 flex items-center self-end sm:self-center">
                        {item.status === "success" ? (
                          <span className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                            Processed
                          </span>
                        ) : (
                          <span
                            className="inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-500/10 text-rose-400 border border-rose-500/20 cursor-help"
                            title={item.details}
                          >
                            Error
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>
        </div>
      </main>
    </div>
  );
}
