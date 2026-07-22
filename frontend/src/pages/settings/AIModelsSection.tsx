import { useState, useEffect } from 'react';
import type { ClaudeModel } from '../../api/types';
import CollapsibleSection from '../../components/CollapsibleSection';
import LoadingSpinner from '../../components/LoadingSpinner';
import { formatModelLabel } from './settingsUtils';

interface AIModelsSectionProps {
  models: ClaudeModel[] | undefined;
  modelsLoading: boolean;
  selectedModel: string;
  verificationModel: string;
  chaptersModel: string;
  onSelectedModelChange: (model: string) => void;
  onVerificationModelChange: (model: string) => void;
  onChaptersModelChange: (model: string) => void;
  onRefresh: () => void;
  refreshIsPending: boolean;
}

function AIModelsSection({
  models,
  modelsLoading,
  selectedModel,
  verificationModel,
  chaptersModel,
  onSelectedModelChange,
  onVerificationModelChange,
  onChaptersModelChange,
  onRefresh,
  refreshIsPending,
}: AIModelsSectionProps) {
  const isOrphan = (value: string) => {
    if (!value || !models) return false;
    return !models.some((m) => m.id === value);
  };

  const [customModel, setCustomModel] = useState(false);
  const [customVerificationModel, setCustomVerificationModel] = useState(false);
  const [customChaptersModel, setCustomChaptersModel] = useState(false);

  // Sync state when models load or if they are orphans
  useEffect(() => {
    if (models) {
      if (isOrphan(selectedModel)) setCustomModel(true);
      if (isOrphan(verificationModel)) setCustomVerificationModel(true);
      if (isOrphan(chaptersModel)) setCustomChaptersModel(true);
    }
  }, [models, selectedModel, verificationModel, chaptersModel]);

  const renderOrphan = (value: string) => {
    if (!value || !models) return null;
    if (models.some((m) => m.id === value)) return null;
    return <option value={value}>{value} (current, not in catalog)</option>;
  };

  return (
    <CollapsibleSection
      title="AI Models"
      defaultOpen
      headerRight={
        <button
          onClick={onRefresh}
          disabled={refreshIsPending}
          className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs rounded bg-secondary text-secondary-foreground hover:bg-secondary/80 disabled:opacity-50 transition-colors"
          title="Refresh model list from provider"
        >
          {refreshIsPending ? (
            <>
              <LoadingSpinner inline className="w-3.5 h-3.5" />
              Refreshing...
            </>
          ) : (
            <>
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </>
          )}
        </button>
      }
    >
      {!modelsLoading && models && models.length === 0 && (
        <div className="mb-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
          <p className="text-sm text-yellow-600 dark:text-yellow-400">
            No models available from the LLM provider. Check that your provider is configured correctly and the endpoint is reachable.
          </p>
        </div>
      )}

      <div className="space-y-4">
        <div>
          <div className="flex justify-between items-center mb-2">
            <label htmlFor="model" className="block text-sm font-medium text-foreground">
              Ad Detection Model
            </label>
            <button
              type="button"
              onClick={() => setCustomModel(!customModel)}
              className="text-xs text-primary hover:underline focus:outline-hidden"
            >
              {customModel ? "Select from list" : "Enter custom model ID"}
            </button>
          </div>
          {customModel ? (
            <input
              type="text"
              id="model"
              value={selectedModel}
              onChange={(e) => onSelectedModelChange(e.target.value)}
              placeholder="e.g. models/gemini-1.5-flash"
              className="w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground focus:outline-hidden focus:ring-2 focus:ring-ring text-sm"
            />
          ) : (
            <select
              id="model"
              value={selectedModel}
              onChange={(e) => onSelectedModelChange(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground focus:outline-hidden focus:ring-2 focus:ring-ring text-sm"
            >
              {renderOrphan(selectedModel)}
              {models?.map((model) => (
                <option key={model.id} value={model.id}>
                  {formatModelLabel(model)}
                </option>
              ))}
            </select>
          )}
          <p className="mt-1 text-sm text-muted-foreground">
            Primary model for analyzing transcripts and detecting ads. Set the model here; the OPENAI_MODEL env var only seeds the initial value on first startup.
          </p>
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <label htmlFor="verificationModel" className="block text-sm font-medium text-foreground">
              Verification Model
            </label>
            <button
              type="button"
              onClick={() => setCustomVerificationModel(!customVerificationModel)}
              className="text-xs text-primary hover:underline focus:outline-hidden"
            >
              {customVerificationModel ? "Select from list" : "Enter custom model ID"}
            </button>
          </div>
          {customVerificationModel ? (
            <input
              type="text"
              id="verificationModel"
              value={verificationModel}
              onChange={(e) => onVerificationModelChange(e.target.value)}
              placeholder="e.g. models/gemini-1.5-flash"
              className="w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground focus:outline-hidden focus:ring-2 focus:ring-ring text-sm"
            />
          ) : (
            <select
              id="verificationModel"
              value={verificationModel}
              onChange={(e) => onVerificationModelChange(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground focus:outline-hidden focus:ring-2 focus:ring-ring text-sm"
            >
              {renderOrphan(verificationModel)}
              {models?.map((model) => (
                <option key={model.id} value={model.id}>
                  {formatModelLabel(model)}
                </option>
              ))}
            </select>
          )}
          <p className="mt-1 text-sm text-muted-foreground">
            Re-runs detection on processed audio to catch missed ads (can differ for cost optimization)
          </p>
        </div>

        <div>
          <div className="flex justify-between items-center mb-2">
            <label htmlFor="chaptersModel" className="block text-sm font-medium text-foreground">
              Chapters Model
            </label>
            <button
              type="button"
              onClick={() => setCustomChaptersModel(!customChaptersModel)}
              className="text-xs text-primary hover:underline focus:outline-hidden"
            >
              {customChaptersModel ? "Select from list" : "Enter custom model ID"}
            </button>
          </div>
          {customChaptersModel ? (
            <input
              type="text"
              id="chaptersModel"
              value={chaptersModel}
              onChange={(e) => onChaptersModelChange(e.target.value)}
              placeholder="e.g. models/gemini-1.5-flash"
              className="w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground focus:outline-hidden focus:ring-2 focus:ring-ring text-sm"
            />
          ) : (
            <select
              id="chaptersModel"
              value={chaptersModel}
              onChange={(e) => onChaptersModelChange(e.target.value)}
              className="w-full px-4 py-2 rounded-lg border border-input bg-background text-foreground focus:outline-hidden focus:ring-2 focus:ring-ring text-sm"
            >
              {renderOrphan(chaptersModel)}
              {models?.map((model) => (
                <option key={model.id} value={model.id}>
                  {formatModelLabel(model)}
                </option>
              ))}
            </select>
          )}
          <p className="mt-1 text-sm text-muted-foreground">
            Chapter title generation and topic detection (smaller/cheaper models work well)
          </p>
        </div>
      </div>
    </CollapsibleSection>
  );
}

export default AIModelsSection;
