// src/styles/theme.ts
// This file provides Tailwind CSS theme constants for consistent usage across the application

export const colors = {
  primary: {
    50: 'bg-primary-50',
    100: 'bg-primary-100',
    200: 'bg-primary-200',
    300: 'bg-primary-300',
    400: 'bg-primary-400',
    500: 'bg-primary-500', // main
    600: 'bg-primary-600',
    700: 'bg-primary-700',
    800: 'bg-primary-800',
    900: 'bg-primary-900',
    text: 'text-primary-500',
    hover: 'hover:bg-primary-600',
    border: 'border-primary-500',
  },
  secondary: {
    50: 'bg-secondary-50',
    100: 'bg-secondary-100',
    200: 'bg-secondary-200',
    300: 'bg-secondary-300',
    400: 'bg-secondary-400',
    500: 'bg-secondary-500', // main
    600: 'bg-secondary-600',
    700: 'bg-secondary-700',
    800: 'bg-secondary-800',
    900: 'bg-secondary-900',
    text: 'text-secondary-500',
    hover: 'hover:bg-secondary-600',
    border: 'border-secondary-500',
  },
  success: {
    50: 'bg-success-50',
    100: 'bg-success-100',
    200: 'bg-success-200',
    300: 'bg-success-300',
    400: 'bg-success-400',
    500: 'bg-success-500', // main
    600: 'bg-success-600',
    700: 'bg-success-700',
    800: 'bg-success-800',
    900: 'bg-success-900',
    text: 'text-success-500',
    hover: 'hover:bg-success-600',
    border: 'border-success-500',
  },
  error: {
    50: 'bg-error-50',
    100: 'bg-error-100',
    200: 'bg-error-200',
    300: 'bg-error-300',
    400: 'bg-error-400',
    500: 'bg-error-500', // main
    600: 'bg-error-600',
    700: 'bg-error-700',
    800: 'bg-error-800',
    900: 'bg-error-900',
    text: 'text-error-500',
    hover: 'hover:bg-error-600',
    border: 'border-error-500',
  },
  warning: {
    50: 'bg-warning-50',
    100: 'bg-warning-100',
    200: 'bg-warning-200',
    300: 'bg-warning-300',
    400: 'bg-warning-400',
    500: 'bg-warning-500', // main
    600: 'bg-warning-600',
    700: 'bg-warning-700',
    800: 'bg-warning-800',
    900: 'bg-warning-900',
    text: 'text-warning-500',
    hover: 'hover:bg-warning-600',
    border: 'border-warning-500',
  },
  info: {
    50: 'bg-info-50',
    100: 'bg-info-100',
    200: 'bg-info-200',
    300: 'bg-info-300',
    400: 'bg-info-400',
    500: 'bg-info-500', // main
    600: 'bg-info-600',
    700: 'bg-info-700',
    800: 'bg-info-800',
    900: 'bg-info-900',
    text: 'text-info-500',
    hover: 'hover:bg-info-600',
    border: 'border-info-500',
  },
  gray: {
    50: 'bg-gray-50',
    100: 'bg-gray-100',
    200: 'bg-gray-200',
    300: 'bg-gray-300',
    400: 'bg-gray-400',
    500: 'bg-gray-500',
    600: 'bg-gray-600',
    700: 'bg-gray-700',
    800: 'bg-gray-800',
    900: 'bg-gray-900',
    text: 'text-gray-500',
    hover: 'hover:bg-gray-600',
    border: 'border-gray-500',
  },
};

export const typography = {
  h1: 'text-4xl font-medium',
  h2: 'text-3xl font-medium',
  h3: 'text-2xl font-medium',
  h4: 'text-xl font-medium',
  h5: 'text-lg font-medium',
  h6: 'text-base font-medium',
  subtitle1: 'text-base font-normal',
  subtitle2: 'text-sm font-medium',
  body1: 'text-base font-normal',
  body2: 'text-sm font-normal',
  button: 'text-sm font-medium',
  caption: 'text-xs font-normal',
  overline: 'text-xs font-normal uppercase',
};

export const spacing = {
  xs: 'p-1',
  sm: 'p-2',
  md: 'p-4',
  lg: 'p-6',
  xl: 'p-8',
};

export const rounded = {
  none: 'rounded-none',
  sm: 'rounded-sm',
  md: 'rounded',
  lg: 'rounded-md',
  xl: 'rounded-lg',
  full: 'rounded-full',
};

export const shadow = {
  none: 'shadow-none',
  sm: 'shadow-sm',
  md: 'shadow',
  lg: 'shadow-md',
  xl: 'shadow-lg',
  '2xl': 'shadow-xl',
};

// Button variants
export const button = {
  primary: `bg-primary-500 hover:bg-primary-600 text-white ${rounded.md} ${spacing.md} font-medium`,
  secondary: `bg-secondary-500 hover:bg-secondary-600 text-white ${rounded.md} ${spacing.md} font-medium`,
  outlined: `border border-primary-500 text-primary-500 hover:bg-primary-50 ${rounded.md} ${spacing.md} font-medium`,
  text: `text-primary-500 hover:bg-primary-50 ${rounded.md} ${spacing.md} font-medium`,
  disabled: 'bg-gray-300 text-gray-500 cursor-not-allowed',
};

// Input variants
export const input = {
  default: 'border border-gray-300 rounded p-2 w-full focus:ring-2 focus:ring-primary-500 focus:border-transparent',
  error: 'border border-error-500 rounded p-2 w-full focus:ring-2 focus:ring-error-500 focus:border-transparent',
  disabled: 'border border-gray-200 bg-gray-100 rounded p-2 w-full text-gray-500 cursor-not-allowed',
};

// Card variants
export const card = {
  default: 'bg-white rounded-lg shadow-md p-4',
  outlined: 'border border-gray-300 rounded-lg p-4',
  elevated: 'bg-white rounded-lg shadow-lg p-4',
};

// Common layout classes
export const layout = {
  container: 'max-w-7xl mx-auto px-4 sm:px-6 lg:px-8',
  section: 'py-12',
  flexRow: 'flex flex-row',
  flexCol: 'flex flex-col',
  flexCenter: 'flex items-center justify-center',
  flexBetween: 'flex items-center justify-between',
  grid: 'grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6',
};

export default {
  colors,
  typography,
  spacing,
  rounded,
  shadow,
  button,
  input,
  card,
  layout,
};
