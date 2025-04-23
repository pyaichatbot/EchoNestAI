import React from 'react';
import { useContent } from '../contexts/ContentContext';
import { useDropzone } from 'react-dropzone';
import { useLanguage, SUPPORTED_LANGUAGES } from '../contexts/LanguageContext';

const ContentUploader: React.FC = () => {
  const { uploadContent, uploadProgress, isLoading } = useContent();
  const { currentLanguage } = useLanguage();
  const [open, setOpen] = React.useState(false);
  const [file, setFile] = React.useState<File | null>(null);
  const [title, setTitle] = React.useState('');
  const [description, setDescription] = React.useState('');
  const [language, setLanguage] = React.useState(currentLanguage);
  const [tags, setTags] = React.useState('');

  // Handle dialog open/close
  const handleOpen = () => setOpen(true);
  const handleClose = () => {
    setOpen(false);
    setFile(null);
    setTitle('');
    setDescription('');
    setLanguage(currentLanguage);
    setTags('');
  };

  // Handle file drop
  const onDrop = React.useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      setFile(file);
      
      // Auto-fill title from filename
      const fileName = file.name.split('.')[0];
      setTitle(fileName.replace(/[_-]/g, ' '));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ 
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/msword': ['.doc'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
      'text/plain': ['.txt'],
      'audio/mpeg': ['.mp3'],
      'audio/wav': ['.wav'],
      'video/mp4': ['.mp4'],
      'video/webm': ['.webm'],
    },
    maxSize: 100 * 1024 * 1024, // 100MB max
    multiple: false
  });

  // Handle language change
  const handleLanguageChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    setLanguage(event.target.value);
  };

  // Handle upload
  const handleUpload = async () => {
    if (!file) return;
    
    // Prepare tags array
    const tagsArray = tags.split(',').map(tag => tag.trim()).filter(tag => tag !== '');
    
    // Upload content
    await uploadContent(file, {
      title,
      description,
      language,
      tags: tagsArray,
    });
    
    // Close dialog
    handleClose();
  };

  // Get file type category
  const getFileType = (file: File): string => {
    const type = file.type;
    if (type.startsWith('audio/')) return 'Audio';
    if (type.startsWith('video/')) return 'Video';
    if (type.startsWith('application/pdf')) return 'PDF';
    if (type.startsWith('application/msword') || type.includes('wordprocessing')) return 'Document';
    if (type.startsWith('text/')) return 'Text';
    return 'File';
  };

  return (
    <>
      <div className="bg-white rounded-lg shadow-md mb-6">
        <div className="p-4">
          <h2 className="text-xl font-medium mb-3">Upload Content</h2>
          <button
            className="bg-primary-500 hover:bg-primary-600 text-white rounded py-3 px-4 w-full flex items-center justify-center"
            onClick={handleOpen}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zM6.293 6.707a1 1 0 010-1.414l3-3a1 1 0 011.414 0l3 3a1 1 0 01-1.414 1.414L11 5.414V13a1 1 0 11-2 0V5.414L7.707 6.707a1 1 0 01-1.414 0z" clipRule="evenodd" />
            </svg>
            Upload New Content
          </button>
        </div>
      </div>

      {/* Upload progress */}
      {uploadProgress && (
        <div className="bg-white rounded-lg shadow-md mb-6">
          <div className="p-4">
            <h2 className="text-xl font-medium mb-3">Upload Progress</h2>
            <div className="w-full mb-3">
              <div className="relative pt-1">
                <div className="h-2 rounded-full bg-gray-200 w-full"></div>
                <div 
                  className="absolute top-0 h-2 rounded-full bg-primary-500 transition-all duration-300"
                  style={{ width: `${uploadProgress.progress}%` }}
                ></div>
              </div>
            </div>
            <p className="text-sm text-gray-600">
              {uploadProgress.status === 'uploading' && `Uploading: ${uploadProgress.progress}%`}
              {uploadProgress.status === 'processing' && 'Processing content...'}
              {uploadProgress.status === 'completed' && 'Upload completed!'}
              {uploadProgress.message && ` - ${uploadProgress.message}`}
            </p>
          </div>
        </div>
      )}

      {/* Upload dialog */}
      {open && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-3xl mx-4">
            <div className="p-6">
              <h2 className="text-xl font-medium mb-4">Upload Content</h2>
              
              <div className="space-y-4">
                <div>
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer transition-colors ${
                      isDragActive ? 'bg-blue-50' : 'hover:bg-blue-50'
                    }`}
                  >
                    <input {...getInputProps()} />
                    {file ? (
                      <div>
                        <p className="text-base">{file.name}</p>
                        <p className="text-sm text-gray-500">
                          {getFileType(file)} - {(file.size / (1024 * 1024)).toFixed(2)} MB
                        </p>
                      </div>
                    ) : (
                      <p>
                        {isDragActive
                          ? 'Drop the file here...'
                          : 'Drag and drop a file here, or click to select a file'}
                      </p>
                    )}
                  </div>
                </div>
                
                <div>
                  <label htmlFor="title" className="block text-sm font-medium text-gray-700 mb-1">
                    Title *
                  </label>
                  <input
                    id="title"
                    type="text"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    required
                  />
                </div>
                
                <div>
                  <label htmlFor="description" className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    id="description"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    rows={3}
                  />
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label htmlFor="language" className="block text-sm font-medium text-gray-700 mb-1">
                      Language
                    </label>
                    <select
                      id="language"
                      value={language}
                      onChange={handleLanguageChange}
                      className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    >
                      {Object.entries(SUPPORTED_LANGUAGES).map(([code, lang]) => (
                        <option key={code} value={code}>
                          {lang.name}
                        </option>
                      ))}
                    </select>
                  </div>
                  
                  <div>
                    <label htmlFor="tags" className="block text-sm font-medium text-gray-700 mb-1">
                      Tags (comma separated)
                    </label>
                    <input
                      id="tags"
                      type="text"
                      value={tags}
                      onChange={(e) => setTags(e.target.value)}
                      className="w-full border border-gray-300 rounded p-2 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      placeholder="education, science, homework"
                    />
                  </div>
                </div>
              </div>
            </div>
            
            <div className="bg-gray-50 px-6 py-3 flex justify-end space-x-2 rounded-b-lg">
              <button
                onClick={handleClose}
                className="px-4 py-2 border border-gray-300 rounded text-gray-700 hover:bg-gray-100"
              >
                Cancel
              </button>
              <button 
                onClick={handleUpload} 
                className={`px-4 py-2 rounded ${
                  !file || !title || isLoading
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-primary-500 hover:bg-primary-600 text-white'
                }`}
                disabled={!file || !title || isLoading}
              >
                Upload
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default ContentUploader;
