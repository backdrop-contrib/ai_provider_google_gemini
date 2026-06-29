<?php

/**
 * @file
 * Google Gemini adapter for the AI module.
 *
 * This adapter implements AIProviderClient using Google's Gemini REST API.
 * No external SDK dependency - uses Backdrop's built-in HTTP functions.
 *
 * @see https://ai.google.dev/docs
 */

class AIGoogleGeminiAdapter extends AIAdapterBase {

  /** @var string */
  protected $baseUrl = 'https://generativelanguage.googleapis.com/v1beta/openai/';

  public function __construct($api_key, ?AIApi $api = NULL) {
    parent::__construct($api_key, $api);
  }

  /**
   * {@inheritdoc}
   *
   * Gemini's native generateContent endpoint authenticates via a URL query
   * parameter (?key=...), so no Authorization header is needed by default.
   * The chat-compatible endpoint (used by chatWithTools) passes the key as
   * a Bearer token via an extra_headers argument to makeRequest().
   */
  protected function getDefaultHeaders(): array {
    return [];
  }

  /**
   * {@inheritdoc}
   *
   * Gemini has real vision-capable models; override the base default.
   */
  public function getModelsByCapability($capability): array {
    $models = $this->getModels();
    if ($capability === 'text') {
      $filtered = $models;
    }
    elseif ($capability === 'vision') {
      $filtered = [];
      foreach ($models as $id => $label) {
        if (preg_match('/gemini-(1\.5|2\.0|3\.1|3-pro)/i', $id)) {
          $filtered[$id] = $label;
        }
      }
    }
    elseif ($capability === 'image') {
      $filtered = $this->getImageModels();
    }
    else {
      $filtered = [];
    }
    backdrop_alter('ai_model_capabilities', $filtered, $capability, $this);
    return $filtered;
  }

  /**
   * {@inheritdoc}
   */
  public function getModels(): array {
    $models = [];
    $image_fallbacks = [
      'nano-banana-2' => 'Nano Banana 2 (Production Visuals)',
      'nano-banana-pro' => 'Nano Banana Pro (Studio 4K)',
      'nano-banana-2.5-flash-image' => 'Nano Banana 2.5 Flash Image',
      'imagen-3.0-generate-002' => 'Imagen 3',
    ];

    try {
      $url = 'https://generativelanguage.googleapis.com/v1beta/models?key=' . urlencode($this->apiKey);
      $options = [
        'method' => 'GET',
        'headers' => [
          'Accept' => 'application/json',
        ],
        'timeout' => 10,
      ];

      $response = backdrop_http_request($url, $options);
      if (isset($response->code) && (int) $response->code === 200) {
        $data = json_decode($response->data, TRUE);
        if (!empty($data['models']) && is_array($data['models'])) {
          foreach ($data['models'] as $item) {
            $name = $item['name'] ?? NULL;
            if (!$name) {
              continue;
            }

            // Strip the "models/" prefix from the ID for a cleaner UI.
            $id = str_replace('models/', '', $name);

            // Filter: Only include models that support content generation or embedding.
            // This reduces the clutter of internal/experimental models in v1beta.
            $methods = $item['supportedGenerationMethods'] ?? [];
            if (!in_array('generateContent', $methods) && !in_array('embedContent', $methods)) {
              continue;
            }

            $display = $item['displayName'] ?? $item['description'] ?? $id;
            $models[$id] = $id . ' — ' . $display;
          }
        }
      }
    }
    catch (\Exception $e) {
      watchdog('ai_provider_google_gemini', 'Failed to fetch Gemini models: @error', ['@error' => $e->getMessage()], WATCHDOG_WARNING);
    }

    // If the API call failed or returned nothing, use fallbacks.
    // If it succeeded, only add fallbacks that aren't already discovered.
    foreach ($image_fallbacks as $id => $label) {
      if (!isset($models[$id])) {
        // Only add if it looks like a model ID we want.
        $models[$id] = $id . ' — ' . $label;
      }
    }

    if (!empty($models)) {
      asort($models);
    }

    return $models;
  }

  /**
   * {@inheritdoc}
   */
  public function completions(string $model, string $prompt, $temperature, $max_tokens = 512, bool $stream_response = FALSE) {
    try {
      // Allow other modules to alter the prompt before sending (e.g., inject site context).
      if (function_exists('backdrop_alter')) {
        $context = [
          'operation' => 'completion',
          'model' => $model,
          'provider' => 'google_gemini',
        ];
        backdrop_alter('ai_prompt', $prompt, $context);
      }

      // Google Gemini doesn't have a separate "completions" endpoint like other chat APIs
      // We use the chat/messages API with a single user message
      $messages = [
        [
          'role' => 'user',
          'content' => trim($prompt),
        ],
      ];

      return $this->chat($model, $messages, $temperature, $max_tokens, $stream_response);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_google_gemini', 'Completions error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function getChatModels(): array {
    return $this->getModelsByCapability('text');
  }

  /**
   * {@inheritdoc}
   */
  public function getImageModels(): array {
    $models = $this->getModels();
    $image_models = [];
    foreach ($models as $id => $label) {
      if (preg_match('/(image|banana|imagen)/i', $id)) {
        $image_models[$id] = $label;
      }
    }
    return $image_models;
  }

  /**
   * {@inheritdoc}
   */
  public function getVisionModels(): array {
    return $this->getModelsByCapability('vision');
  }

  /**
   * {@inheritdoc}
   */
  public function getEmbeddingModels(): array {
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function getModerationModels(): array {
    return [];
  }

  /**
   * {@inheritdoc}
   */
  public function chat(string $model, array $messages, $temperature, $max_tokens = 1024, bool $stream_response = FALSE, array $context_extra = []) {
    try {
      // Allow other modules to alter chat messages before sending (e.g., inject site context).
      if (function_exists('backdrop_alter')) {
        $context = [
          'operation' => 'chat',
          'model' => $model,
          'provider' => 'google_gemini',
        ];
        if (empty($context_extra['skip_ai_message_alter'])) {
          backdrop_alter('ai_chat_messages', $messages, $context);
        }
      }

      // Convert messages to Gemini-native format
      $gemini_contents = $this->convertMessagesGemini($messages);
      $body = [
        'contents' => $gemini_contents,
        'generationConfig' => [
          'maxOutputTokens' => (int) $max_tokens ?: 1024,
          'temperature' => (float) $temperature,
        ],
      ];
      // The native generateContent API has no 'system' message role;
      // convertMessagesGemini() skips them, so route system messages into
      // the top-level systemInstruction field instead of dropping them.
      $system_texts = [];
      foreach ($messages as $msg) {
        if (($msg['role'] ?? '') !== 'system') {
          continue;
        }
        $content = $msg['content'] ?? '';
        if (is_array($content)) {
          foreach ($content as $block) {
            if (is_string($block)) {
              $system_texts[] = $block;
            }
            elseif (is_array($block) && ($block['type'] ?? '') === 'text') {
              $system_texts[] = (string) ($block['text'] ?? '');
            }
          }
        }
        else {
          $system_texts[] = (string) $content;
        }
      }
      if ($system_texts) {
        $body['systemInstruction'] = [
          'parts' => [['text' => implode("\n\n", $system_texts)]],
        ];
      }
      // Normalize the incoming model identifier into the Gemini resource
      // format the API expects (e.g. "models/gemini-1.5-pro"). The model
      // string may come in several shapes depending on the higher-level UI
      // (e.g. "google_gemini/models/gemini-1.5-pro", "models/gemini-1.5-pro",
      // or just "gemini-1.5-pro"). Normalize to a single "models/..." form
      // and then append to the endpoint path.
      $resource = $model;
      // If the string contains 'models/' anywhere, take that segment onward.
      if (strpos($resource, 'models/') !== FALSE) {
        $resource = substr($resource, strpos($resource, 'models/'));
      }
      elseif (strpos($resource, '/') !== FALSE) {
        // If other prefix exists (e.g., 'provider/model'), take the last segment.
        $resource = substr($resource, strrpos($resource, '/') + 1);
        $resource = 'models/' . $resource;
      }
      else {
        // Bare id (e.g., 'gemini-1.5-pro') -> prefix with models/.
        $resource = 'models/' . $resource;
      }

      $url = 'https://generativelanguage.googleapis.com/v1beta/' . $resource . ':generateContent?key=' . urlencode($this->apiKey);
      if ($stream_response) {
        // Gemini streaming not implemented in this adapter yet
        throw new \Exception('Streaming not implemented for Gemini-native API');
      }

      // Primary attempt: call the per-model generateContent endpoint.
      // Log the normalized resource and body keys for debugging.
      watchdog('ai_provider_google_gemini', 'Gemini generateContent primary attempt for @resource with keys: @keys', [
        '@resource' => $resource,
        '@keys' => implode(',', array_keys($body)),
      ], WATCHDOG_DEBUG);

      try {
        // Long generations can exceed the 30s makeRequest() default.
        $response = $this->makeRequest($url, $body, [], 'POST', 300);
      }
      catch (\Exception $e) {
        $msg = $e->getMessage();
        // Detect a broader set of model/payload errors returned by Gemini.
        if (preg_match('/(unexpected model name format|GenerateContentRequest\.model|Invalid JSON payload|Unknown name)/i', $msg)) {
          watchdog('ai_provider_google_gemini', 'Model format/payload error for @resource: @msg; retrying with alternate endpoint', ['@resource' => $resource, '@msg' => $msg], WATCHDOG_WARNING);

          $alt_url = 'https://generativelanguage.googleapis.com/v1beta/models:generateContent?key=' . urlencode($this->apiKey);
          // Include explicit model field using the canonical resource.
          $body_with_model = $body;
          $body_with_model['model'] = $resource;

          try {
            watchdog('ai_provider_google_gemini', 'Retrying Gemini generateContent via models:generateContent with model=@m', ['@m' => $resource], WATCHDOG_DEBUG);
            $response = $this->makeRequest($alt_url, $body_with_model, [], 'POST', 300);
          }
          catch (\Exception $e2) {
            // As a last-ditch attempt, try the bare model id (without 'models/').
            $bare = preg_replace('#^models/#i', '', $resource);
            $body_with_model['model'] = $bare;
            try {
              watchdog('ai_provider_google_gemini', 'Retrying Gemini generateContent with bare model=@m', ['@m' => $bare], WATCHDOG_DEBUG);
              $response = $this->makeRequest($alt_url, $body_with_model, [], 'POST', 300);
            }
            catch (\Exception $e3) {
              // Log all three failures and rethrow the last exception.
              watchdog('ai_provider_google_gemini', 'All Gemini generateContent attempts failed for @resource. Errors: primary=@err1 alt=@err2 last=@err3', ['@resource' => $resource, '@err1' => $msg, '@err2' => $e2->getMessage(), '@err3' => $e3->getMessage()], WATCHDOG_ERROR);
              throw $e3;
            }
          }
        }
        else {
          throw $e;
        }
      }

      // Extract text from Gemini response
      $text = $this->extractCandidateText($response);
      return trim($text);
    }
    catch (\Exception $e) {
      watchdog('ai_provider_google_gemini', 'Chat error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Extract the primary candidate text from a Gemini response.
   *
   * Handles several response shapes safely and concatenates multiple parts
   * when present.
   *
   * @param array|null $response
   *   Decoded response array from Gemini API.
   *
   * @return string
   *   The extracted text or empty string when none found.
   */
  protected function extractCandidateText($response): string {
    if (empty($response) || !is_array($response)) {
      return '';
    }

    // Look for candidates -> content -> parts -> text
    if (!empty($response['candidates']) && is_array($response['candidates'])) {
      $first = $response['candidates'][0];
      if (!empty($first['content'])) {
        $content = $first['content'];
        // If content.parts is present, join any 'text' entries.
        if (!empty($content['parts']) && is_array($content['parts'])) {
          $pieces = [];
          foreach ($content['parts'] as $part) {
            if (is_string($part) && trim($part) !== '') {
              $pieces[] = $part;
            }
            elseif (is_array($part) && isset($part['text'])) {
              $pieces[] = $part['text'];
            }
          }
          if (!empty($pieces)) {
            return implode("\n", $pieces);
          }
        }
        // Some variants may put text directly under content
        if (isset($content['text']) && is_string($content['text'])) {
          return $content['text'];
        }
      }
      // Fallback: sometimes candidate may contain a top-level 'text' key
      if (isset($first['text']) && is_string($first['text'])) {
        return $first['text'];
      }
    }

    // Fallback: check for other common shapes such as 'output' or 'message'.
    if (!empty($response['output']) && is_string($response['output'])) {
      return $response['output'];
    }
    if (!empty($response['message']) && is_string($response['message'])) {
      return $response['message'];
    }

    return '';
  }

  /**
   * {@inheritdoc}
   */
  public function images(string $model, string $prompt, string $size, string $response_format, string $quality = 'standard', string $style = 'natural', ?string $output_format = NULL) {
    try {
      // Normalize model name.
      $resource = $model;
      if (strpos($resource, 'models/') !== FALSE) {
        $resource = substr($resource, strpos($resource, 'models/'));
      }
      elseif (strpos($resource, '/') !== FALSE) {
        $resource = substr($resource, strrpos($resource, '/') + 1);
        $resource = 'models/' . $resource;
      }
      else {
        $resource = 'models/' . $resource;
      }

      // Gemini 3 / Nano Banana models use the same generateContent endpoint for images.
      // The prompt is sent as text, and the response contains inlineData with the image.
      $url = 'https://generativelanguage.googleapis.com/v1beta/' . $resource . ':generateContent?key=' . urlencode($this->apiKey);

      $body = [
        'contents' => [
          [
            'role' => 'user',
            'parts' => [
              ['text' => $prompt],
            ],
          ],
        ],
        'generationConfig' => [
          'candidateCount' => 1,
        ],
      ];

      // Image generation regularly takes longer than 30 seconds.
      $response = $this->makeRequest($url, $body, [], 'POST', 300);

      $images = [];
      if (!empty($response['candidates'][0]['content']['parts'])) {
        foreach ($response['candidates'][0]['content']['parts'] as $part) {
          if (!empty($part['inlineData'])) {
            $mime_type = $part['inlineData']['mimeType'] ?? 'image/png';
            $base64_data = $part['inlineData']['data'];

            if ($response_format === 'b64_json') {
              $images[] = ['b64_json' => $base64_data];
            }
            else {
              $images[] = [
                'url' => 'data:' . $mime_type . ';base64,' . $base64_data,
              ];
            }
          }
        }
      }

      if (empty($images)) {
        throw new \Exception('No image data returned from Gemini API. Ensure the model supports image generation.');
      }

      return [
        'created' => time(),
        'data' => $images,
      ];
    }
    catch (\Exception $e) {
      watchdog('ai_provider_google_gemini', 'Image generation error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * {@inheritdoc}
   */
  public function textToSpeech(string $model, string $input, string $voice, string $response_format) {
    watchdog('ai_provider_google_gemini',
      'Text-to-speech is not supported by Google Gemini adapter.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Text-to-speech is not supported by Google Gemini adapter.');
  }

  /**
   * {@inheritdoc}
   */
  public function speechToText(string $model, string $file, string $task = 'transcribe', $temperature = 0.4, string $response_format = 'verbose_json') {
    watchdog('ai_provider_google_gemini',
      'Speech-to-text is not supported by Google Gemini adapter.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Speech-to-text is not supported by Google Gemini adapter.');
  }

  /**
   * {@inheritdoc}
   */
  public function moderation(string $input, string $model = 'gemini-moderation'): array {
    watchdog('ai_provider_google_gemini',
      'Moderation API is not directly available in Gemini. Gemini has built-in safety filters.',
      [], WATCHDOG_WARNING);
    throw new \RuntimeException('Moderation is not supported by Google Gemini adapter.');
  }

  /**
   * {@inheritdoc}
   *
   * Backwards-compatible single-input embedding wrapper that calls the bulk
   * embeddings() helper and returns the first embedding result in the same
   * predictable shape as other adapters (object/data/index).
   */
  public function embedding(string $input, string $model, bool $log = TRUE): array {
    $start_time = microtime(TRUE);
    try {
      // Reuse the bulk embeddings method with a single-item array.
      $result = $this->embeddings($model, [$input]);
      if (!empty($result['data']) && is_array($result['data']) && isset($result['data'][0])) {
        if (isset($this->api) && method_exists($this->api, 'recordLog')) {
          $duration = microtime(TRUE) - $start_time;
          $this->api->recordLog('embedding', $model, ['input' => $input], $result, TRUE, $duration, NULL, !$log);
        }
        // Return only the vector to match the new AIProviderClient::embedding contract
        return is_array($result['data'][0]) ? ($result['data'][0]['embedding'] ?? []) : [];
      }
      throw new \RuntimeException('Embedding request returned no embedding data.');
    }
    catch (\Exception $e) {
      if (isset($this->api) && method_exists($this->api, 'recordLog')) {
        $duration = microtime(TRUE) - $start_time;
        $this->api->recordLog('embedding', $model, ['input' => $input], NULL, FALSE, $duration, $e->getMessage(), !$log);
      }
      ai_log_embedding_error('ai_provider_google_gemini', $e->getMessage(), $log);
      return [];
    }
  }

  /**
   * {@inheritdoc}
   */
  public function embeddings(string $model, array $inputs, string $response_format = 'float'): array {
    if ($response_format !== 'float') {
      throw new \InvalidArgumentException('AIGoogleGeminiAdapter::embeddings() only supports response_format "float"; "' . $response_format . '" is not supported.');
    }
    try {
      // Use the native embedContent endpoint; the body below is the native
      // shape (content.parts.text), not the OpenAI-compatible one.
      $embedding_model = $model ?: 'gemini-embedding-2';
      if (strpos($embedding_model, 'models/') !== FALSE) {
        $embedding_model = substr($embedding_model, strpos($embedding_model, 'models/') + strlen('models/'));
      }
      elseif (strpos($embedding_model, '/') !== FALSE) {
        $embedding_model = substr($embedding_model, strrpos($embedding_model, '/') + 1);
      }
      $results = [];

      $url = 'https://generativelanguage.googleapis.com/v1beta/models/' . $embedding_model . ':batchEmbedContents?key=' . urlencode($this->apiKey);

      $requests = [];
      foreach ($inputs as $input) {
        $requests[] = [
          'model' => 'models/' . $embedding_model,
          'content' => [
            'parts' => [
              ['text' => $input],
            ],
          ],
        ];
      }

      $response = $this->makeRequest($url, ['requests' => $requests], [], 'POST', 300);

      $seq = 0;
      foreach ($inputs as $index => $input) {
        if (!isset($response['embeddings'][$seq]['values']) || !is_array($response['embeddings'][$seq]['values'])) {
          throw new \Exception('Gemini batchEmbedContents response missing vector for input index ' . $index . ' (request position ' . $seq . ')');
        }
        $results[] = [
          'object' => 'embedding',
          'embedding' => $response['embeddings'][$seq]['values'],
          'index' => $index,
        ];
        $seq++;
      }

      return [
        'object' => 'list',
        'data' => $results,
        'model' => $embedding_model,
        'usage' => [
          'prompt_tokens' => count($inputs),
          'total_tokens' => count($inputs),
        ],
      ];
    }
    catch (\Exception $e) {
      watchdog('ai_provider_google_gemini', 'Embeddings error: @error',
        ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }

  /**
   * Convert provider-neutral chat-style messages to Gemini-native format.
   *
   * @param array $messages
   *   Array of messages in the shared chat format.
   *
   * @return array
   *   Converted messages for Gemini API.
   */
  protected function convertMessagesGemini(array $messages): array {
    $gemini_contents = [];
    foreach ($messages as $msg) {
      $role = $msg['role'] ?? 'user';
      $gemini_role = ($role === 'assistant') ? 'model' : 'user';
      if ($role === 'system') {
        continue;
      }
      $content = $msg['content'] ?? '';
      $parts = [];
      if (is_array($content)) {
        foreach ($content as $block) {
          $parts[] = $this->convertContentBlockGemini($block);
        }
      }
      else {
        $parts[] = ['text' => (string) $content];
      }
      $gemini_contents[] = [
        'role' => $gemini_role,
        'parts' => $parts,
      ];
    }
    return $gemini_contents;
  }

  /**
   * Convert a shared chat content block to a Gemini part.
   */
  protected function convertContentBlockGemini($block): array {
    if (is_string($block)) {
      return ['text' => $block];
    }
    if (!is_array($block)) {
      return ['text' => json_encode($block)];
    }

    $type = $block['type'] ?? '';
    if ($type === 'text') {
      return ['text' => (string) ($block['text'] ?? '')];
    }

    if ($type === 'image_url') {
      $image_url = $block['image_url']['url'] ?? '';
      if (preg_match('/^data:([^;]+);base64,(.+)$/', $image_url, $matches)) {
        return [
          'inline_data' => [
            'mime_type' => $matches[1],
            'data' => $matches[2],
          ],
        ];
      }
      return ['text' => 'Unsupported Gemini image URL format.'];
    }

    return ['text' => json_encode($block)];
  }

  /**
   * Make HTTP request to Gemini API.
   *
   * @param string $url
   *   The API endpoint URL.
   * @param array $body
   *   The request body.
   *
   * @return array
   *   The parsed JSON response.
   */
  protected function makeRequest(string $url, array $body = [], array $extra_headers = [], string $method = 'POST', int $timeout = 30): array {
    $options = [
      'method' => strtoupper($method),
      'headers' => array_merge([
        'Content-Type' => 'application/json',
      ], $extra_headers),
      'timeout' => $timeout,
    ];

    if ($options['method'] !== 'GET') {
      $options['data'] = json_encode($body);
    }

    $response = backdrop_http_request($url, $options);

    $code = isset($response->code) ? (int) $response->code : 0;
    $body_text = isset($response->data) ? $response->data : '';

    // Mask API key in URLs for logging.
    $masked_url = preg_replace('/(key=)[^&\s]+/i', '$1***', $url);
    $log_body_keys = is_array($body) ? implode(',', array_keys($body)) : '';

    // On non-2xx responses, log a masked diagnostic without raw body text.
    if ($code < 200 || $code >= 300) {
      watchdog('ai_provider_google_gemini', 'Gemini request failed. url=@url; code=@code; body_keys=@keys; response_length=@length', [
        '@url' => $masked_url,
        '@code' => $code,
        '@keys' => $log_body_keys,
        '@length' => is_string($body_text) ? strlen($body_text) : 0,
      ], WATCHDOG_WARNING);
    }

    // Treat any 2xx as success.
    if ($code >= 200 && $code < 300) {
      if (is_array($body_text) || is_object($body_text)) {
        // JSON round-trip normalizes nested stdClass objects to arrays.
        return json_decode(json_encode($body_text), TRUE) ?: [];
      }
      $raw_ok = (string) $body_text;
      if ($raw_ok === '') {
        return [];
      }
      $decoded_ok = json_decode($raw_ok, TRUE);
      if (!is_array($decoded_ok)) {
        throw new \Exception('Gemini API returned non-JSON body for HTTP ' . $code . ': ' . json_last_error_msg());
      }
      return $decoded_ok;
    }

    // For non-2xx responses, attempt to decode the body. Some Gemini
    // responses can include a success-shaped payload even when the status
    // is unexpected; if so, treat them as success to avoid false errors.
    $decoded = NULL;
    // If the returned data is already an array/object (some wrappers do
    // this), normalize it to an array.
    if (is_array($body_text)) {
      $decoded = $body_text;
    }
    elseif (is_object($body_text)) {
      $decoded = (array) $body_text;
    }
    else {
      // Attempt normal JSON decode first.
      $decoded = json_decode((string) $body_text, TRUE);

      // If decode failed but the raw text contains recognizable keys,
      // try extracting the JSON object substring and decode that.
      if ($decoded === NULL) {
        $raw = (string) $body_text;
        if (stripos($raw, 'candidates') !== FALSE || stripos($raw, 'modelVersion') !== FALSE) {
          // Find a balanced JSON object around the anchor (e.g., "candidates").
          $anchor_pos = stripos($raw, 'candidates');
          if ($anchor_pos === FALSE) {
            $anchor_pos = stripos($raw, 'modelVersion');
          }
          if ($anchor_pos !== FALSE) {
            // Find the last '{' before the anchor.
            $start = NULL;
            for ($i = $anchor_pos; $i >= 0; $i--) {
              if ($raw[$i] === '{') { $start = $i; break; }
            }
            if ($start !== NULL) {
              $depth = 0;
              $end = NULL;
              $len = strlen($raw);
              for ($j = $start; $j < $len; $j++) {
                if ($raw[$j] === '{') { $depth++; }
                elseif ($raw[$j] === '}') { $depth--; }
                if ($depth === 0) { $end = $j; break; }
              }
              if ($end !== NULL && $end > $start) {
                $try = substr($raw, $start, $end - $start + 1);
                $decoded = json_decode($try, TRUE);
              }
            }
          }
         }
       }
     }

     if (is_array($decoded) && (!empty($decoded['candidates']) || !empty($decoded['modelVersion']) || !empty($decoded['usageMetadata']))) {
      watchdog('ai_provider_google_gemini', 'Gemini returned non-2xx but success-looking payload (code=@code): @keys', ['@code' => $code, '@keys' => implode(',', array_keys($decoded))], WATCHDOG_WARNING);
      return $decoded;
    }

    // As a last resort: if the raw body contains candidate-like text but we
    // couldn't parse it, synthesize a minimal response so higher-level code
    // can extract the text instead of failing completely. This avoids
    // crashing when providers return slightly malformed/extra-wrapped JSON.
    $raw = is_string($body_text) ? $body_text : (is_scalar($body_text) ? (string) $body_text : json_encode($body_text));
    if (stripos($raw, 'candidates') !== FALSE || stripos($raw, 'modelVersion') !== FALSE || stripos($raw, 'usageMetadata') !== FALSE) {
      watchdog('ai_provider_google_gemini', 'Gemini returned candidate-like payload but could not decode JSON; returning synthetic candidate.', [], WATCHDOG_WARNING);
      return [
        'candidates' => [
          [
            'content' => [
              'parts' => [
                ['text' => $raw],
              ],
            ],
          ],
        ],
      ];
    }

    // Prefer the parsed Gemini error.message to avoid leaking raw body content.
    $parsed = is_string($body_text) ? json_decode($body_text, TRUE) : (is_array($body_text) ? $body_text : NULL);
    if (isset($parsed['error']['message']) && is_string($parsed['error']['message'])) {
      $error = $parsed['error']['message'];
    }
    elseif (!empty($response->error)) {
      // Transport failures (timeout, DNS, TLS) come back as code -1 with an
      // empty body; the cURL/socket message lives in ->error.
      $error = trim((string) $response->error);
    }
    else {
      $body_len = is_string($body_text) ? strlen($body_text) : 0;
      $error = 'unexpected response (body length: ' . $body_len . ')';
    }
    throw new \Exception('Gemini API error (' . $code . '): ' . $error);
  }

  /**
   * Handle streaming responses for Gemini.
   *
   * @param string $url
   *   The API endpoint URL.
   * @param array $body
   *   The request body.
   *
   * @return mixed
   *   A streaming HTTP response.
   */
  protected function handleStreamingRequest($url, array $body) {
    $options = [
      'method' => 'POST',
      'headers' => [
        'Accept' => 'text/event-stream',
        'Content-Type' => 'application/json',
        'Authorization' => 'Bearer ' . $this->apiKey,
      ],
      'data' => json_encode($body),
      'timeout' => 300,
    ];
    $options['stream'] = TRUE;

    return new AIStreamingResponse($url, $options, function ($data) {
      return $data['choices'][0]['delta']['content'] ?? NULL;
    });
  }

  /**
   * {@inheritdoc}
   *
   * Gemini supports function calling via its chat-compatible endpoint,
   * so we pass tools through the standard Chat Completions payload.
   */
  public function chatWithTools(string $model, array $messages, array $tools, $temperature, $max_tokens = 1024, string $tool_choice = 'auto', array $context_extra = []): array {
    try {
      // Normalize model ID: strip provider prefix and "models/" for the chat-compatible endpoint.
      $id = $model;
      if (strpos($id, '/') !== FALSE) {
        $id = substr($id, strrpos($id, '/') + 1);
      }
      $id = str_replace('models/', '', $id);

      $payload = [
        'model'       => $id,
        'messages'    => $messages,
        'tools'       => $tools,
        'tool_choice' => $tool_choice,
        'temperature' => (float) $temperature,
      ];
      if ((int) $max_tokens > 0) {
        $payload['max_tokens'] = (int) $max_tokens;
      }

      $url = $this->baseUrl . 'chat/completions';
      try {
        $result = $this->makeRequest($url, $payload, ['Authorization' => 'Bearer ' . $this->apiKey], 'POST', 300);
      }
      catch (\Exception $e) {
        watchdog('ai_provider_google_gemini', 'chatWithTools HTTP error: @msg', ['@msg' => $e->getMessage()], WATCHDOG_ERROR);
        throw $e;
      }
      $choice = $result['choices'][0] ?? [];
      $message = $choice['message'] ?? [];
      $finish_reason = $choice['finish_reason'] ?? 'stop';
      $content = trim($message['content'] ?? '');
      $tool_calls = [];
      foreach ($message['tool_calls'] ?? [] as $tc) {
        $args = $tc['function']['arguments'] ?? '{}';
        if (is_string($args)) {
          $args = json_decode($args, TRUE) ?? [];
        }
        $tool_calls[] = [
          'id'        => $tc['id'] ?? '',
          'name'      => $tc['function']['name'] ?? '',
          'arguments' => $args,
        ];
      }
      return [
        'finish_reason' => $finish_reason,
        'content'       => $content,
        'tool_calls'    => $tool_calls,
        'raw'           => $result,
      ];
    }
    catch (\Exception $e) {
      watchdog('ai_provider_google_gemini', 'chatWithTools error: @error', ['@error' => $e->getMessage()], WATCHDOG_ERROR);
      throw $e;
    }
  }
}
