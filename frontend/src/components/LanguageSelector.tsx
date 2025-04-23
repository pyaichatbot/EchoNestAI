import React from 'react';
import { useLanguage, SUPPORTED_LANGUAGES } from '../contexts/LanguageContext';

const LanguageSelector: React.FC = () => {
  const { currentLanguage, changeLanguage, supportedLanguages } = useLanguage();

  const handleChange = (event: React.ChangeEvent<HTMLSelectElement>) => {
    changeLanguage(event.target.value);
  };

  return (
    <div className="min-w-[120px]">
      <div className="relative">
        <select
          value={currentLanguage}
          onChange={handleChange}
          className="w-full appearance-none bg-transparent text-white border border-white/50 hover:border-white/80 rounded py-1 pl-3 pr-8 text-sm focus:outline-none focus:ring-2 focus:ring-white/70"
        >
          {Object.entries(supportedLanguages).map(([code, language]) => (
            <option key={code} value={code} className="bg-gray-800 text-white">
              {language.name}
            </option>
          ))}
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-white">
          <svg className="h-4 w-4 fill-current" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20">
            <path d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z" />
          </svg>
        </div>
      </div>
    </div>
  );
};

export default LanguageSelector;
