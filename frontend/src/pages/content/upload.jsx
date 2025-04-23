import React, { useState, useEffect } from 'react';
import Layout from '../../components/Layout';
import { useContent } from '../../contexts/ContentContext';
import { useLanguage } from '../../contexts/LanguageContext';

const ContentUploadPage = () => {
  const { uploadContent } = useContent();
  const { translate, currentLanguage, languages } = useLanguage();
  const [file, setFile] = useState(null);
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [language, setLanguage] = useState(currentLanguage);
  const [contentType, setContentType] = useState('document');
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadStatus, setUploadStatus] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const [eventSource, setEventSource] = useState(null);

  // Clean up event source on unmount
  useEffect(() => {
    return () => {
      if (eventSource) {
        eventSource.close();
      }
    };
  }, [eventSource]);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      
      // Auto-detect content type from file extension
      const extension = selectedFile.name.split('.').pop().toLowerCase();
      if (['pdf', 'doc', 'docx', 'txt', 'md'].includes(extension)) {
        setContentType('document');
      } else if (['mp3', 'wav', 'ogg', 'm4a'].includes(extension)) {
        setContentType('audio');
      } else if (['mp4', 'mov', 'avi', 'webm'].includes(extension)) {
        setContentType('video');
      }
      
      // Auto-set title from filename if not already set
      if (!title) {
        const fileName = selectedFile.name.split('.')[0];
        setTitle(fileName.replace(/[_-]/g, ' '));
      }
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!file || !title) {
      setUploadStatus({
        type: 'error',
        message: 'Please provide a file and title'
      });
      return;
    }
    
    setIsUploading(true);
    setUploadProgress(0);
    setUploadStatus({
      type: 'info',
      message: 'Starting upload...'
    });
    
    try {
      const contentId = await uploadContent(
        file, 
        {
          title,
          description,
          language,
          type: contentType
        },
        (progress) => {
          setUploadProgress(progress);
        }
      );
      
      if (contentId) {
        setUploadStatus({
          type: 'success',
          message: 'Upload successful!'
        });
        
        // Connect to SSE for processing updates
        connectToProcessingEvents(contentId);
      } else {
        setUploadStatus({
          type: 'error',
          message: 'Upload failed. Please try again.'
        });
      }
    } catch (err) {
      console.error('Error uploading content:', err);
      setUploadStatus({
        type: 'error',
        message: `Upload failed: ${err.message}`
      });
    } finally {
      setIsUploading(false);
    }
  };

  const connectToProcessingEvents = (contentId) => {
    const token = localStorage.getItem('token');
    const sse = new EventSource(
      `${process.env.NEXT_PUBLIC_API_URL}/api/v1/content/${contentId}/processing-events?token=${token}`
    );
    
    setProcessingStatus({
      type: 'info',
      message: 'Processing content...',
      progress: 0
    });
    
    sse.onmessage = (event) => {
      const data = JSON.parse(event.data);
      
      setProcessingStatus({
        type: data.status === 'completed' ? 'success' : 'info',
        message: data.message,
        progress: data.progress || 0
      });
      
      if (data.status === 'completed' || data.status === 'failed') {
        sse.close();
      }
    };
    
    sse.onerror = () => {
      setProcessingStatus({
        type: 'warning',
        message: 'Lost connection to processing updates. The process will continue in the background.',
        progress: processingStatus?.progress || 0
      });
      sse.close();
    };
    
    setEventSource(sse);
  };

  const resetForm = () => {
    setFile(null);
    setTitle('');
    setDescription('');
    setLanguage(currentLanguage);
    setContentType('document');
    setUploadProgress(0);
    setUploadStatus(null);
    setProcessingStatus(null);
    
    // Reset file input
    const fileInput = document.getElementById('file-upload');
    if (fileInput) {
      fileInput.value = '';
    }
  };

  return (
    <Layout>
      <div className="py-6">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8">
          <div className="flex justify-between items-center">
            <h1 className="text-2xl font-semibold text-secondary">Upload Content</h1>
            <a
              href="/content"
              className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
            >
              Back to Content Library
            </a>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 md:px-8 mt-6">
          <div className="bg-white shadow overflow-hidden sm:rounded-lg">
            <div className="px-4 py-5 sm:p-6">
              <form onSubmit={handleSubmit}>
                <div className="space-y-6">
                  {/* File upload */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Content File
                    </label>
                    <div className="mt-1 flex justify-center px-6 pt-5 pb-6 border-2 border-gray-300 border-dashed rounded-md">
                      <div className="space-y-1 text-center">
                        <svg className="mx-auto h-12 w-12 text-gray-400" stroke="currentColor" fill="none" viewBox="0 0 48 48" aria-hidden="true">
                          <path d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m4-24h8m-4-4v8m-12 4h.02" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                        </svg>
                        <div className="flex text-sm text-gray-600">
                          <label htmlFor="file-upload" className="relative cursor-pointer bg-white rounded-md font-medium text-primary hover:text-primary-dark focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-primary">
                            <span>Upload a file</span>
                            <input id="file-upload" name="file-upload" type="file" className="sr-only" onChange={handleFileChange} />
                          </label>
                          <p className="pl-1">or drag and drop</p>
                        </div>
                        <p className="text-xs text-gray-500">
                          PDF, DOCX, TXT, MP3, MP4, etc. up to 100MB
                        </p>
                        {file && (
                          <p className="text-sm text-primary mt-2">
                            Selected: {file.name} ({(file.size / 1024 / 1024).toFixed(2)} MB)
                          </p>
                        )}
                      </div>
                    </div>
                  </div>

                  {/* Content type */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Content Type
                    </label>
                    <div className="mt-1">
                      <select
                        value={contentType}
                        onChange={(e) => setContentType(e.target.value)}
                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary focus:border-primary sm:text-sm rounded-md"
                      >
                        <option value="document">Document</option>
                        <option value="audio">Audio</option>
                        <option value="video">Video</option>
                      </select>
                    </div>
                  </div>

                  {/* Title */}
                  <div>
                    <label htmlFor="title" className="block text-sm font-medium text-gray-700">
                      Title
                    </label>
                    <div className="mt-1">
                      <input
                        type="text"
                        name="title"
                        id="title"
                        value={title}
                        onChange={(e) => setTitle(e.target.value)}
                        className="shadow-sm focus:ring-primary focus:border-primary block w-full sm:text-sm border-gray-300 rounded-md"
                        placeholder="Content title"
                        required
                      />
                    </div>
                  </div>

                  {/* Description */}
                  <div>
                    <label htmlFor="description" className="block text-sm font-medium text-gray-700">
                      Description
                    </label>
                    <div className="mt-1">
                      <textarea
                        id="description"
                        name="description"
                        rows={3}
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                        className="shadow-sm focus:ring-primary focus:border-primary block w-full sm:text-sm border-gray-300 rounded-md"
                        placeholder="Brief description of the content"
                      />
                    </div>
                  </div>

                  {/* Language */}
                  <div>
                    <label className="block text-sm font-medium text-gray-700">
                      Language
                    </label>
                    <div className="mt-1">
                      <select
                        value={language}
                        onChange={(e) => setLanguage(e.target.value)}
                        className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-primary focus:border-primary sm:text-sm rounded-md"
                      >
                        {languages.map((lang) => (
                          <option key={lang.code} value={lang.code}>
                            {lang.name}
                          </option>
                        ))}
                      </select>
                    </div>
                  </div>

                  {/* Upload status */}
                  {uploadStatus && (
                    <div className={`rounded-md p-4 ${
                      uploadStatus.type === 'error' ? 'bg-red-50' :
                      uploadStatus.type === 'success' ? 'bg-green-50' :
                      'bg-blue-50'
                    }`}>
                      <div className="flex">
                        <div className="flex-shrink-0">
                          {uploadStatus.type === 'error' && (
                            <svg className="h-5 w-5 text-red-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                          )}
                          {uploadStatus.type === 'success' && (
                            <svg className="h-5 w-5 text-green-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                          )}
                          {uploadStatus.type === 'info' && (
                            <svg className="h-5 w-5 text-blue-400" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor">
                              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                            </svg>
                          )}
                        </div>
                        <div className="ml-3">
                          <p className={`text-sm font-medium ${
                            uploadStatus.type === 'error' ? 'text-red-800' :
                            uploadStatus.type === 'success' ? 'text-green-800' :
                            'text-blue-800'
                          }`}>
                            {uploadStatus.message}
                          </p>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Upload progress */}
                  {isUploading && (
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-primary">Uploading</span>
                        <span className="text-sm font-medium text-primary">{uploadProgress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div className="bg-primary h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }}></div>
                      </div>
                    </div>
                  )}

                  {/* Processing status */}
                  {processingStatus && (
                    <div>
                      <div className="flex justify-between mb-1">
                        <span className="text-sm font-medium text-primary">Processing</span>
                        <span className="text-sm font-medium text-primary">{processingStatus.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2.5">
                        <div className="bg-primary h-2.5 rounded-full" style={{ width: `${processingStatus.progress}%` }}></div>
                      </div>
                      <p className="mt-1 text-sm text-gray-500">{processingStatus.message}</p>
                    </div>
                  )}

                  {/* Submit buttons */}
                  <div className="flex justify-end space-x-3">
                    <button
                      type="button"
                      onClick={resetForm}
                      className="inline-flex items-center px-4 py-2 border border-gray-300 shadow-sm text-sm font-medium rounded-md text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary"
                      disabled={isUploading}
                    >
                      Reset
                    </button>
                    <button
                      type="submit"
                      className="inline-flex items-center px-4 py-2 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-primary hover:bg-primary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary disabled:opacity-50"
                      disabled={isUploading || !file}
                    >
                      {isUploading ? (
                        <>
                          <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                          </svg>
                          Uploading...
                        </>
                      ) : (
                        'Upload Content'
                      )}
                    </button>
                  </div>
                </div>
              </form>
            </div>
          </div>
        </div>
      </div>
    </Layout>
  );
};

export default ContentUploadPage;
