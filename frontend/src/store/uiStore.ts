/**
 * Argus Core - UI Store
 * =====================
 * Global UI state management using Zustand.
 * 
 * Implements: PRIME_FRONTEND_DOCUMENT.md - Section 2.2 - store/uiStore.ts
 * 
 * Role: Manage global UI state including modals, sidebars, notifications,
 * and other cross-cutting UI concerns.
 * 
 * Integration:
 * - Used by: Layout components, modal triggers, notification system
 * - Persists: Theme preference (via ThemeProvider separately)
 */

import { create } from 'zustand';
import { devtools } from 'zustand/middleware';

// ============== TYPES ==============

/**
 * Modal types that can be shown
 */
export type ModalType = 
  | 'analysis-options'
  | 'delete-confirm'
  | 'report-preview'
  | 'help'
  | 'settings'
  | null;

/**
 * Toast notification severity
 */
export type ToastSeverity = 'info' | 'success' | 'warning' | 'error';

/**
 * Toast notification data
 */
export interface Toast {
  id: string;
  title: string;
  message?: string;
  severity: ToastSeverity;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

/**
 * Sidebar state
 */
export interface SidebarState {
  isOpen: boolean;
  isCollapsed: boolean;
}

/**
 * UI Store state interface
 */
interface UIState {
  // Modal state
  activeModal: ModalType;
  modalData: Record<string, unknown> | null;
  
  // Sidebar state
  sidebar: SidebarState;
  
  // Toast notifications
  toasts: Toast[];
  
  // Loading states
  isPageLoading: boolean;
  loadingMessage: string | null;
  
  // Mobile menu
  isMobileMenuOpen: boolean;
  
  // Command palette
  isCommandPaletteOpen: boolean;
}

/**
 * UI Store actions interface
 */
interface UIActions {
  // Modal actions
  openModal: (type: ModalType, data?: Record<string, unknown>) => void;
  closeModal: () => void;
  
  // Sidebar actions
  toggleSidebar: () => void;
  setSidebarOpen: (isOpen: boolean) => void;
  toggleSidebarCollapse: () => void;
  
  // Toast actions
  addToast: (toast: Omit<Toast, 'id'>) => void;
  removeToast: (id: string) => void;
  clearToasts: () => void;
  
  // Loading actions
  setPageLoading: (isLoading: boolean, message?: string) => void;
  
  // Mobile menu actions
  toggleMobileMenu: () => void;
  setMobileMenuOpen: (isOpen: boolean) => void;
  
  // Command palette actions
  toggleCommandPalette: () => void;
  setCommandPaletteOpen: (isOpen: boolean) => void;
  
  // Reset
  resetUI: () => void;
}

// ============== INITIAL STATE ==============

const initialState: UIState = {
  activeModal: null,
  modalData: null,
  sidebar: {
    isOpen: true,
    isCollapsed: false,
  },
  toasts: [],
  isPageLoading: false,
  loadingMessage: null,
  isMobileMenuOpen: false,
  isCommandPaletteOpen: false,
};

// ============== STORE ==============

/**
 * UI Store with Zustand
 * Manages global UI state with devtools support
 */
export const useUIStore = create<UIState & UIActions>()(
  devtools(
    (set, get) => ({
      // Initial state
      ...initialState,
      
      // ============== MODAL ACTIONS ==============
      
      /**
       * Open a modal with optional data
       */
      openModal: (type, data = null) => {
        set(
          { activeModal: type, modalData: data },
          false,
          'openModal'
        );
      },
      
      /**
       * Close the active modal
       */
      closeModal: () => {
        set(
          { activeModal: null, modalData: null },
          false,
          'closeModal'
        );
      },
      
      // ============== SIDEBAR ACTIONS ==============
      
      /**
       * Toggle sidebar open/closed
       */
      toggleSidebar: () => {
        set(
          (state) => ({
            sidebar: {
              ...state.sidebar,
              isOpen: !state.sidebar.isOpen,
            },
          }),
          false,
          'toggleSidebar'
        );
      },
      
      /**
       * Set sidebar open state directly
       */
      setSidebarOpen: (isOpen) => {
        set(
          (state) => ({
            sidebar: {
              ...state.sidebar,
              isOpen,
            },
          }),
          false,
          'setSidebarOpen'
        );
      },
      
      /**
       * Toggle sidebar collapsed state (icon-only mode)
       */
      toggleSidebarCollapse: () => {
        set(
          (state) => ({
            sidebar: {
              ...state.sidebar,
              isCollapsed: !state.sidebar.isCollapsed,
            },
          }),
          false,
          'toggleSidebarCollapse'
        );
      },
      
      // ============== TOAST ACTIONS ==============
      
      /**
       * Add a toast notification
       */
      addToast: (toast) => {
        const id = `toast-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
        const newToast: Toast = {
          ...toast,
          id,
          duration: toast.duration ?? 5000,
        };
        
        set(
          (state) => ({
            toasts: [...state.toasts, newToast],
          }),
          false,
          'addToast'
        );
        
        // Auto-remove toast after duration
        if (newToast.duration && newToast.duration > 0) {
          setTimeout(() => {
            get().removeToast(id);
          }, newToast.duration);
        }
      },
      
      /**
       * Remove a toast by ID
       */
      removeToast: (id) => {
        set(
          (state) => ({
            toasts: state.toasts.filter((t) => t.id !== id),
          }),
          false,
          'removeToast'
        );
      },
      
      /**
       * Clear all toasts
       */
      clearToasts: () => {
        set({ toasts: [] }, false, 'clearToasts');
      },
      
      // ============== LOADING ACTIONS ==============
      
      /**
       * Set page loading state
       */
      setPageLoading: (isLoading, message = null) => {
        set(
          {
            isPageLoading: isLoading,
            loadingMessage: isLoading ? message : null,
          },
          false,
          'setPageLoading'
        );
      },
      
      // ============== MOBILE MENU ACTIONS ==============
      
      /**
       * Toggle mobile menu
       */
      toggleMobileMenu: () => {
        set(
          (state) => ({ isMobileMenuOpen: !state.isMobileMenuOpen }),
          false,
          'toggleMobileMenu'
        );
      },
      
      /**
       * Set mobile menu open state
       */
      setMobileMenuOpen: (isOpen) => {
        set({ isMobileMenuOpen: isOpen }, false, 'setMobileMenuOpen');
      },
      
      // ============== COMMAND PALETTE ACTIONS ==============
      
      /**
       * Toggle command palette
       */
      toggleCommandPalette: () => {
        set(
          (state) => ({ isCommandPaletteOpen: !state.isCommandPaletteOpen }),
          false,
          'toggleCommandPalette'
        );
      },
      
      /**
       * Set command palette open state
       */
      setCommandPaletteOpen: (isOpen) => {
        set({ isCommandPaletteOpen: isOpen }, false, 'setCommandPaletteOpen');
      },
      
      // ============== RESET ==============
      
      /**
       * Reset UI to initial state
       */
      resetUI: () => {
        set(initialState, false, 'resetUI');
      },
    }),
    {
      name: 'argus-ui-store',
      enabled: process.env.NODE_ENV === 'development',
    }
  )
);

// ============== SELECTORS ==============

/**
 * Selector for modal state
 */
export const selectModal = (state: UIState & UIActions) => ({
  activeModal: state.activeModal,
  modalData: state.modalData,
  openModal: state.openModal,
  closeModal: state.closeModal,
});

/**
 * Selector for sidebar state
 */
export const selectSidebar = (state: UIState & UIActions) => ({
  sidebar: state.sidebar,
  toggleSidebar: state.toggleSidebar,
  setSidebarOpen: state.setSidebarOpen,
  toggleSidebarCollapse: state.toggleSidebarCollapse,
});

/**
 * Selector for toasts
 */
export const selectToasts = (state: UIState & UIActions) => ({
  toasts: state.toasts,
  addToast: state.addToast,
  removeToast: state.removeToast,
  clearToasts: state.clearToasts,
});

/**
 * Selector for loading state
 */
export const selectLoading = (state: UIState & UIActions) => ({
  isPageLoading: state.isPageLoading,
  loadingMessage: state.loadingMessage,
  setPageLoading: state.setPageLoading,
});

export default useUIStore;
