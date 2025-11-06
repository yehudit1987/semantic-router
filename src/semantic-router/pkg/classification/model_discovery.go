package classification

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// ModelPaths holds the discovered model paths
type ModelPaths struct {
	// Legacy ModernBERT models (low confidence)
	ModernBertBase     string
	IntentClassifier   string
	PIIClassifier      string
	SecurityClassifier string

	// LoRA models
	LoRAIntentClassifier   string
	LoRAPIIClassifier      string
	LoRASecurityClassifier string
	LoRAArchitecture       string // "bert", "roberta", "modernbert"
}

// IsComplete checks if all required models are found
func (mp *ModelPaths) IsComplete() bool {
	return mp.HasLoRAModels() || mp.HasLegacyModels()
}

// HasLoRAModels checks if LoRA models are available
func (mp *ModelPaths) HasLoRAModels() bool {
	return mp.LoRAIntentClassifier != "" &&
		mp.LoRAPIIClassifier != "" &&
		mp.LoRASecurityClassifier != "" &&
		mp.LoRAArchitecture != ""
}

// HasLegacyModels checks if legacy ModernBERT models are available
func (mp *ModelPaths) HasLegacyModels() bool {
	return mp.ModernBertBase != "" &&
		mp.IntentClassifier != "" &&
		mp.PIIClassifier != "" &&
		mp.SecurityClassifier != ""
}

// PreferLoRA returns true if LoRA models should be used (higher confidence)
func (mp *ModelPaths) PreferLoRA() bool {
	return mp.HasLoRAModels()
}

// ArchitectureModels holds models for a specific architecture
type ArchitectureModels struct {
	Intent   string
	PII      string
	Security string
}

// AutoDiscoverModels automatically discovers model files in the models directory
// Uses intelligent architecture selection: BERT > RoBERTa > ModernBERT
func AutoDiscoverModels(modelsDir string) (*ModelPaths, error) {
	if modelsDir == "" {
		modelsDir = "./models"
	}

	// Check if models directory exists
	if _, err := os.Stat(modelsDir); os.IsNotExist(err) {
		return nil, fmt.Errorf("models directory does not exist: %s", modelsDir)
	}

	// Collect all available LoRA models by architecture
	architectureModels := map[string]*ArchitectureModels{
		"bert":       {Intent: "", PII: "", Security: ""},
		"roberta":    {Intent: "", PII: "", Security: ""},
		"modernbert": {Intent: "", PII: "", Security: ""},
	}

	// Legacy models for fallback
	legacyPaths := &ModelPaths{}

	// Walk through the models directory to collect all models
	err := filepath.Walk(modelsDir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Skip files, we're looking for directories
		if !info.IsDir() {
			return nil
		}

		dirName := strings.ToLower(info.Name())

		// Collect LoRA models by architecture
		switch {
		case strings.HasPrefix(dirName, "lora_intent_classifier"):
			arch := detectArchitectureFromPath(dirName)
			if architectureModels[arch].Intent == "" {
				architectureModels[arch].Intent = path
			}

		case strings.HasPrefix(dirName, "lora_pii_detector"):
			arch := detectArchitectureFromPath(dirName)
			if architectureModels[arch].PII == "" {
				architectureModels[arch].PII = path
			}

		case strings.HasPrefix(dirName, "lora_jailbreak_classifier"):
			arch := detectArchitectureFromPath(dirName)
			if architectureModels[arch].Security == "" {
				architectureModels[arch].Security = path
			}

		// Legacy ModernBERT models (fallback for backward compatibility)
		case strings.Contains(dirName, "modernbert") && strings.Contains(dirName, "base") && !strings.Contains(dirName, "classifier"):
			// ModernBERT base model: "modernbert-base", "modernbert_base", etc.
			if legacyPaths.ModernBertBase == "" {
				legacyPaths.ModernBertBase = path
			}

		case strings.Contains(dirName, "category") && strings.Contains(dirName, "classifier"):
			if legacyPaths.IntentClassifier == "" {
				legacyPaths.IntentClassifier = path
			}
			if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				legacyPaths.ModernBertBase = path
			}

		case strings.Contains(dirName, "pii") && strings.Contains(dirName, "classifier"):
			if legacyPaths.PIIClassifier == "" {
				legacyPaths.PIIClassifier = path
			}
			if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				legacyPaths.ModernBertBase = path
			}

		case (strings.Contains(dirName, "jailbreak") || strings.Contains(dirName, "security")) && strings.Contains(dirName, "classifier"):
			if legacyPaths.SecurityClassifier == "" {
				legacyPaths.SecurityClassifier = path
			}
			if legacyPaths.ModernBertBase == "" && strings.Contains(dirName, "modernbert") {
				legacyPaths.ModernBertBase = path
			}
		}

		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("error scanning models directory: %w", err)
	}

	// Intelligent architecture selection based on performance priority: BERT > RoBERTa > ModernBERT
	architecturePriority := []string{"bert", "roberta", "modernbert"}

	fmt.Printf("[ModelDiscovery] Starting architecture selection...\n")
	fmt.Printf("[ModelDiscovery] Found architectures: BERT(Intent=%v, PII=%v, Security=%v), RoBERTa(Intent=%v, PII=%v, Security=%v), ModernBERT(Intent=%v, PII=%v, Security=%v)\n",
		architectureModels["bert"].Intent != "",
		architectureModels["bert"].PII != "",
		architectureModels["bert"].Security != "",
		architectureModels["roberta"].Intent != "",
		architectureModels["roberta"].PII != "",
		architectureModels["roberta"].Security != "",
		architectureModels["modernbert"].Intent != "",
		architectureModels["modernbert"].PII != "",
		architectureModels["modernbert"].Security != "")

	for _, arch := range architecturePriority {
		models := architectureModels[arch]
		fmt.Printf("[ModelDiscovery] Checking %s architecture...\n", arch)

		// Check if this architecture has a complete set of models
		if models.Intent != "" && models.PII != "" && models.Security != "" {
			fmt.Printf("[ModelDiscovery] %s has all three model types present\n", arch)

			// Validate that each model directory is complete and has required files
			// This is critical in CI where models might be partially downloaded
			loraChecks := map[string]string{
				"LoRA Intent classifier":   models.Intent,
				"LoRA PII classifier":      models.PII,
				"LoRA Security classifier": models.Security,
			}

			// Check if all models in this architecture are valid
			architectureValid := true
			for name, path := range loraChecks {
				fmt.Printf("[ModelDiscovery] Validating %s at %s\n", name, path)
				if err := validateModelDirectory(path, name); err != nil {
					fmt.Printf("[ModelDiscovery] ❌ %s validation failed: %v\n", name, err)
					// This architecture is incomplete/invalid, skip to next
					architectureValid = false
					break
				}
				fmt.Printf("[ModelDiscovery] ✅ %s validation passed\n", name)
			}

			// If this architecture is fully valid, use it
			if architectureValid {
				fmt.Printf("[ModelDiscovery] ✅ Selected %s architecture (all models valid)\n", arch)
				return &ModelPaths{
					LoRAIntentClassifier:   models.Intent,
					LoRAPIIClassifier:      models.PII,
					LoRASecurityClassifier: models.Security,
					LoRAArchitecture:       arch,
					// Copy legacy paths for fallback compatibility
					ModernBertBase:     legacyPaths.ModernBertBase,
					IntentClassifier:   legacyPaths.IntentClassifier,
					PIIClassifier:      legacyPaths.PIIClassifier,
					SecurityClassifier: legacyPaths.SecurityClassifier,
				}, nil
			}
			fmt.Printf("[ModelDiscovery] ❌ Skipping %s architecture (validation failed)\n", arch)
		} else {
			fmt.Printf("[ModelDiscovery] ❌ Skipping %s architecture (incomplete set: Intent=%v, PII=%v, Security=%v)\n",
				arch, models.Intent != "", models.PII != "", models.Security != "")
		}
	}

	fmt.Printf("[ModelDiscovery] ⚠️  No complete LoRA architecture found, falling back to legacy models\n")

	// If no complete LoRA architecture set found, return legacy models
	return legacyPaths, nil
}

// detectArchitectureFromPath detects model architecture from directory name
func detectArchitectureFromPath(dirName string) string {
	switch {
	case strings.Contains(dirName, "bert-base-uncased"):
		return "bert"
	case strings.Contains(dirName, "roberta-base"):
		return "roberta"
	case strings.Contains(dirName, "modernbert-base"):
		return "modernbert"
	default:
		// Default fallback
		return "bert"
	}
}

// ValidateModelPaths validates that all discovered paths contain valid model files
func ValidateModelPaths(paths *ModelPaths) error {
	if paths == nil {
		return fmt.Errorf("model paths is nil")
	}

	// If LoRA models are available, validate them
	if paths.HasLoRAModels() {
		loraChecks := map[string]string{
			"LoRA Intent classifier":   paths.LoRAIntentClassifier,
			"LoRA PII classifier":      paths.LoRAPIIClassifier,
			"LoRA Security classifier": paths.LoRASecurityClassifier,
		}

		for name, path := range loraChecks {
			if path == "" {
				return fmt.Errorf("%s model not found", name)
			}

			// Check if directory exists and contains model files
			if err := validateModelDirectory(path, name); err != nil {
				return err
			}
		}
		return nil
	}

	// If no LoRA models, validate legacy models
	if paths.HasLegacyModels() {
		legacyChecks := map[string]string{
			"ModernBERT base":     paths.ModernBertBase,
			"Intent classifier":   paths.IntentClassifier,
			"PII classifier":      paths.PIIClassifier,
			"Security classifier": paths.SecurityClassifier,
		}

		for name, path := range legacyChecks {
			if path == "" {
				return fmt.Errorf("%s model not found", name)
			}

			// Check if directory exists and contains model files
			if err := validateModelDirectory(path, name); err != nil {
				return err
			}
		}
		return nil
	}

	return fmt.Errorf("no valid models found (neither LoRA nor legacy)")
}

// validateModelDirectory checks if a directory contains valid model files
func validateModelDirectory(path, modelName string) error {
	// Check if directory exists
	info, err := os.Stat(path)
	if os.IsNotExist(err) {
		return fmt.Errorf("%s model directory does not exist: %s", modelName, path)
	}
	if !info.IsDir() {
		return fmt.Errorf("%s model path is not a directory: %s", modelName, path)
	}

	// Check for common model files (at least one should exist)
	commonModelFiles := []string{
		"config.json",
		"pytorch_model.bin",
		"model.safetensors",
		"tokenizer.json",
		"vocab.txt",
	}

	hasModelFile := false
	for _, filename := range commonModelFiles {
		if _, err := os.Stat(filepath.Join(path, filename)); err == nil {
			hasModelFile = true
			break
		}
	}

	if !hasModelFile {
		return fmt.Errorf("%s model directory appears to be empty or invalid: %s", modelName, path)
	}

	// Additional validation for LoRA models - require lora_config.json
	if strings.Contains(strings.ToLower(modelName), "lora") {
		loraConfigPath := filepath.Join(path, "lora_config.json")
		if _, err := os.Stat(loraConfigPath); os.IsNotExist(err) {
			return fmt.Errorf("%s is missing required lora_config.json file: %s", modelName, path)
		}
	}

	return nil
}

// GetModelDiscoveryInfo returns detailed information about model discovery
func GetModelDiscoveryInfo(modelsDir string) map[string]interface{} {
	info := map[string]interface{}{
		"models_directory":  modelsDir,
		"discovery_status":  "failed",
		"discovered_models": map[string]interface{}{},
		"missing_models":    []string{},
		"errors":            []string{},
	}

	paths, err := AutoDiscoverModels(modelsDir)
	if err != nil {
		info["errors"] = append(info["errors"].([]string), err.Error())
		return info
	}

	// Add discovered models
	discovered := map[string]interface{}{
		"modernbert_base":     paths.ModernBertBase,
		"intent_classifier":   paths.IntentClassifier,
		"pii_classifier":      paths.PIIClassifier,
		"security_classifier": paths.SecurityClassifier,
	}
	info["discovered_models"] = discovered

	// Check for missing models
	missing := []string{}
	if paths.ModernBertBase == "" {
		missing = append(missing, "ModernBERT base model")
	}
	if paths.IntentClassifier == "" {
		missing = append(missing, "Intent classifier")
	}
	if paths.PIIClassifier == "" {
		missing = append(missing, "PII classifier")
	}
	if paths.SecurityClassifier == "" {
		missing = append(missing, "Security classifier")
	}
	info["missing_models"] = missing

	// Validate discovered models
	if err := ValidateModelPaths(paths); err != nil {
		info["errors"] = append(info["errors"].([]string), err.Error())
		info["discovery_status"] = "incomplete"
	} else {
		info["discovery_status"] = "complete"
	}

	return info
}

// AutoInitializeUnifiedClassifier attempts to auto-discover and initialize the unified classifier
// Prioritizes LoRA models over legacy ModernBERT models
func AutoInitializeUnifiedClassifier(modelsDir string) (*UnifiedClassifier, error) {
	fmt.Printf("[AutoInit] Starting unified classifier initialization from %s\n", modelsDir)

	// Discover models
	paths, err := AutoDiscoverModels(modelsDir)
	if err != nil {
		return nil, fmt.Errorf("model discovery failed: %w", err)
	}

	// Validate paths
	if err := ValidateModelPaths(paths); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}

	// Check if we should use LoRA models
	if paths.PreferLoRA() {
		fmt.Printf("[AutoInit] ✅ Using LoRA models (architecture: %s)\n", paths.LoRAArchitecture)
		fmt.Printf("[AutoInit]   Intent:   %s\n", paths.LoRAIntentClassifier)
		fmt.Printf("[AutoInit]   PII:      %s\n", paths.LoRAPIIClassifier)
		fmt.Printf("[AutoInit]   Security: %s\n", paths.LoRASecurityClassifier)
		return initializeLoRAUnifiedClassifier(paths)
	}

	// Fallback to legacy ModernBERT initialization
	fmt.Printf("[AutoInit] ⚠️  Falling back to legacy ModernBERT models\n")
	fmt.Printf("[AutoInit]   Intent:   %s\n", paths.IntentClassifier)
	fmt.Printf("[AutoInit]   PII:      %s\n", paths.PIIClassifier)
	fmt.Printf("[AutoInit]   Security: %s\n", paths.SecurityClassifier)
	return initializeLegacyUnifiedClassifier(paths)
}

// initializeLoRAUnifiedClassifier initializes with LoRA models
func initializeLoRAUnifiedClassifier(paths *ModelPaths) (*UnifiedClassifier, error) {
	fmt.Printf("[LoRAInit] Creating unified classifier with LoRA mode\n")

	// Create unified classifier instance with LoRA mode
	classifier := &UnifiedClassifier{
		initialized: false,
		useLoRA:     true, // Mark as LoRA mode for high confidence
	}

	// Store LoRA model paths for later initialization
	// The actual C initialization will be done in unified_classifier.go
	classifier.loraModelPaths = &LoRAModelPaths{
		IntentPath:   paths.LoRAIntentClassifier,
		PIIPath:      paths.LoRAPIIClassifier,
		SecurityPath: paths.LoRASecurityClassifier,
		Architecture: paths.LoRAArchitecture,
	}

	// Mark as initialized - the actual C initialization will be lazy-loaded
	classifier.initialized = true

	fmt.Printf("[LoRAInit] Pre-initializing LoRA C bindings...\n")
	// Pre-initialize LoRA C bindings to avoid lazy loading during first API call
	if err := classifier.initializeLoRABindings(); err != nil {
		return nil, fmt.Errorf("failed to pre-initialize LoRA bindings: %w", err)
	}
	classifier.loraInitialized = true
	fmt.Printf("[LoRAInit] ✅ LoRA classifier initialized successfully\n")

	return classifier, nil
}

// initializeLegacyUnifiedClassifier initializes with legacy ModernBERT models
func initializeLegacyUnifiedClassifier(paths *ModelPaths) (*UnifiedClassifier, error) {
	fmt.Printf("[LegacyInit] Initializing with legacy ModernBERT models\n")
	// Load intent labels from the actual model's mapping file
	categoryMappingPath := filepath.Join(paths.IntentClassifier, "category_mapping.json")
	categoryMapping, err := LoadCategoryMapping(categoryMappingPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load category mapping from %s: %w", categoryMappingPath, err)
	}

	// Extract intent labels in correct order (by index)
	intentLabels := make([]string, len(categoryMapping.IdxToCategory))
	for i := 0; i < len(categoryMapping.IdxToCategory); i++ {
		if label, exists := categoryMapping.IdxToCategory[fmt.Sprintf("%d", i)]; exists {
			intentLabels[i] = label
		} else {
			return nil, fmt.Errorf("missing label for index %d in category mapping", i)
		}
	}

	// Load PII labels from the actual model's mapping file
	var piiLabels []string
	piiMappingPath := filepath.Join(paths.PIIClassifier, "pii_type_mapping.json")
	if _, statErr := os.Stat(piiMappingPath); statErr == nil {
		piiMapping, loadErr := LoadPIIMapping(piiMappingPath)
		if loadErr != nil {
			return nil, fmt.Errorf("failed to load PII mapping from %s: %w", piiMappingPath, loadErr)
		}
		// Extract labels from PII mapping (ordered by index)
		piiLabels = make([]string, len(piiMapping.IdxToLabel))
		for i := 0; i < len(piiMapping.IdxToLabel); i++ {
			if label, exists := piiMapping.IdxToLabel[fmt.Sprintf("%d", i)]; exists {
				piiLabels[i] = label
			} else {
				return nil, fmt.Errorf("missing PII label for index %d", i)
			}
		}
	} else {
		return nil, fmt.Errorf("PII mapping file not found at %s - required for unified classifier", piiMappingPath)
	}

	// Load security labels from the actual model's mapping file
	var securityLabels []string
	securityMappingPath := filepath.Join(paths.SecurityClassifier, "jailbreak_type_mapping.json")
	if _, statErr2 := os.Stat(securityMappingPath); statErr2 == nil {
		jailbreakMapping, loadErr2 := LoadJailbreakMapping(securityMappingPath)
		if loadErr2 != nil {
			return nil, fmt.Errorf("failed to load jailbreak mapping from %s: %w", securityMappingPath, loadErr2)
		}
		// Extract labels from jailbreak mapping (ordered by index)
		securityLabels = make([]string, len(jailbreakMapping.IdxToLabel))
		for i := 0; i < len(jailbreakMapping.IdxToLabel); i++ {
			if label, exists := jailbreakMapping.IdxToLabel[fmt.Sprintf("%d", i)]; exists {
				securityLabels[i] = label
			} else {
				return nil, fmt.Errorf("missing security label for index %d", i)
			}
		}
	} else {
		return nil, fmt.Errorf("security mapping file not found at %s - required for unified classifier", securityMappingPath)
	}

	// Get global unified classifier instance
	classifier := GetGlobalUnifiedClassifier()

	// Check if already initialized (can happen in tests or if called multiple times)
	if classifier.IsInitialized() {
		return classifier, nil
	}

	// Initialize with discovered paths and config-based labels
	err = classifier.Initialize(
		paths.ModernBertBase,
		paths.IntentClassifier,
		paths.PIIClassifier,
		paths.SecurityClassifier,
		intentLabels,
		piiLabels,
		securityLabels,
		false, // Default to GPU, will fallback to CPU if needed
	)
	if err != nil {
		return nil, fmt.Errorf("unified classifier initialization failed: %w", err)
	}

	return classifier, nil
}
