# Design Document: AI Multi-Modal Interview Assistant

## Overview

The AI Multi-Modal Interview Assistant is a sophisticated interview practice platform that combines multiple AI technologies to provide comprehensive feedback to students. The system architecture follows a microservices-inspired approach with clear separation between frontend, backend API, real-time communication, and specialized analysis services.

### High-Level Architecture

The system consists of the following major components:

1. **Frontend Application** (React.js): Single-page application providing user interface
2. **API Server** (Express.js): RESTful API handling authentication, session management, and data operations
3. **Real-Time Server** (Socket.io): WebSocket server managing live interview sessions
4. **RAG Engine**: Adaptive question generation using vector search and LLM
5. **Analysis Pipeline**: Multimodal processing including speech, facial, and eye tracking analysis
6. **Storage Layer**: MongoDB Atlas, Redis cache, and AWS S3 for different data types
7. **Report Generator**: PDF generation service for post-interview reports

### Technology Stack Rationale

- **React.js + Tailwind CSS**: Modern, component-based UI with utility-first styling
- **Express.js**: Lightweight, flexible Node.js framework for API development
- **Socket.io**: Reliable real-time bidirectional communication
- **MongoDB Atlas**: Document database with native vector search capabilities
- **Redis**: High-performance caching layer
- **AWS S3**: Scalable object storage for media files
- **Python Services**: TensorFlow, librosa, OpenCV for ML/CV tasks
- **Google Gemini**: State-of-the-art embeddings and language generation

## Architecture

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (React.js)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │ Interview UI │  │  Dashboard   │  │  Session History   │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
         │                    │                      │
         │ HTTPS/REST         │ WebSocket           │ HTTPS/REST
         ▼                    ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Backend Services                            │
│  ┌──────────────────┐         ┌─────────────────────────┐      │
│  │   API Server     │◄────────┤  Real-Time Server       │      │
│  │  (Express.js)    │         │    (Socket.io)          │      │
│  └──────────────────┘         └─────────────────────────┘      │
│         │                              │                         │
│         │                              │                         │
│  ┌──────▼──────────────────────────────▼───────────────┐       │
│  │           Session Orchestrator                       │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
         │                    │                      │
         ▼                    ▼                      ▼
┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  RAG Engine  │    │ Analysis Pipeline│    │ Report Generator│
│              │    │                  │    │                 │
│ ┌──────────┐ │    │ ┌──────────────┐│    │ ┌─────────────┐ │
│ │ Gemini   │ │    │ │Speech Analyzer││   │ │PDF Generator│ │
│ │Embeddings│ │    │ └──────────────┘│    │ └─────────────┘ │
│ └──────────┘ │    │ ┌──────────────┐│    └─────────────────┘
│ ┌──────────┐ │    │ │Facial Analyzer││
│ │ Vector   │ │    │ └──────────────┘│
│ │ Search   │ │    │ ┌──────────────┐│
│ └──────────┘ │    │ │ Eye Tracker  ││
│ ┌──────────┐ │    │ └──────────────┘│
│ │ Gemini   │ │    │ ┌──────────────┐│
│ │ 2.0 Flash│ │    │ │Fusion Engine ││
│ └──────────┘ │    │ └──────────────┘│
└──────────────┘    └──────────────────┘
         │                    │
         ▼                    ▼
┌─────────────────────────────────────────┐
│          Storage Layer                   │
│  ┌──────────┐  ┌───────┐  ┌──────────┐ │
│  │ MongoDB  │  │ Redis │  │  AWS S3  │ │
│  │  Atlas   │  │ Cache │  │  Media   │ │
│  └──────────┘  └───────┘  └──────────┘ │
└─────────────────────────────────────────┘
```

### Component Interaction Flow


**Interview Session Flow:**

1. Student authenticates → JWT token issued
2. Student starts session → Session record created in MongoDB
3. Frontend establishes WebSocket connection → Real-time server ready
4. RAG Engine generates first question → Sent via WebSocket
5. Student responds (audio + video) → Streams sent to backend
6. Analysis Pipeline processes multimodal data in parallel
7. Fusion Engine combines results → Feedback sent to frontend
8. RAG Engine evaluates answer → Generates next question (adaptive)
9. Cycle repeats until session ends
10. Report Generator creates PDF → Stored in S3
11. Session data persisted → MongoDB + S3

## Components and Interfaces

### 1. Frontend Application

**Technology:** React.js, Tailwind CSS, WebRTC, MediaRecorder API

**Responsibilities:**
- User authentication and session management
- Interview UI with video preview and question display
- Real-time feedback visualization
- Media capture (audio/video) using WebRTC
- Session history and analytics display
- PDF report download

**Key Components:**

```typescript
// Authentication
interface AuthService {
  login(email: string, password: string): Promise<AuthToken>
  logout(): void
  refreshToken(): Promise<AuthToken>
  getCurrentUser(): User | null
}

// Session Management
interface SessionManager {
  startSession(config: SessionConfig): Promise<Session>
  endSession(sessionId: string): Promise<SessionSummary>
  pauseSession(sessionId: string): void
  resumeSession(sessionId: string): void
}

// Media Capture
interface MediaCapture {
  startAudioRecording(): Promise<MediaStream>
  startVideoRecording(): Promise<MediaStream>
  stopRecording(): void
  getAudioChunks(): Blob[]
  getVideoFrames(): ImageData[]
}

// Real-time Communication
interface RealtimeClient {
  connect(token: string): Promise<void>
  disconnect(): void
  sendAnswer(audio: Blob, text: string): void
  onQuestion(callback: (question: Question) => void): void
  onFeedback(callback: (feedback: Feedback) => void): void
}
```


### 2. API Server (Express.js)

**Responsibilities:**
- User authentication and authorization (JWT)
- Session CRUD operations
- Session history retrieval
- Report download endpoints
- Employer analytics endpoints
- Rate limiting and input validation

**API Endpoints:**

```typescript
// Authentication
POST   /api/auth/register        // Register new user
POST   /api/auth/login           // Login and get JWT
POST   /api/auth/refresh         // Refresh JWT token
POST   /api/auth/logout          // Invalidate token

// Sessions
POST   /api/sessions             // Create new session
GET    /api/sessions             // Get user's session history
GET    /api/sessions/:id         // Get specific session details
DELETE /api/sessions/:id         // Delete session
GET    /api/sessions/:id/report  // Download PDF report

// Analytics (Student)
GET    /api/analytics/progress   // Get progress over time
GET    /api/analytics/strengths  // Get identified strengths
GET    /api/analytics/weaknesses // Get areas for improvement

// Analytics (Employer)
GET    /api/employer/dashboard   // Get aggregated metrics
GET    /api/employer/trends      // Get performance trends
GET    /api/employer/top-performers // Get top students

// Media
POST   /api/media/upload         // Upload media chunk
GET    /api/media/:id/url        // Get pre-signed S3 URL
```

**Middleware Stack:**

```typescript
interface Middleware {
  authenticate: (req, res, next) => void    // Verify JWT
  authorize: (roles: string[]) => Middleware // Check user role
  validateInput: (schema: Schema) => Middleware // Validate request body
  rateLimit: (config: RateLimitConfig) => Middleware // Rate limiting
  errorHandler: (err, req, res, next) => void // Global error handling
}
```


### 3. Real-Time Server (Socket.io)

**Responsibilities:**
- WebSocket connection management
- Real-time question delivery
- Real-time feedback streaming
- Audio/video chunk reception
- Session state synchronization

**Socket Events:**

```typescript
// Client → Server Events
interface ClientEvents {
  'session:start': (config: SessionConfig) => void
  'session:end': () => void
  'answer:audio': (chunk: ArrayBuffer) => void
  'answer:text': (text: string) => void
  'answer:complete': () => void
}

// Server → Client Events
interface ServerEvents {
  'session:ready': (sessionId: string) => void
  'question:new': (question: Question) => void
  'feedback:realtime': (metrics: RealtimeMetrics) => void
  'feedback:answer': (evaluation: AnswerEvaluation) => void
  'session:complete': (summary: SessionSummary) => void
  'error': (error: ErrorMessage) => void
}

// Real-time Metrics
interface RealtimeMetrics {
  timestamp: number
  confidenceScore: number      // 0-100
  attentionScore: number        // 0-100
  currentEmotion: EmotionState
  speechPace: 'too_fast' | 'optimal' | 'too_slow'
  eyeContact: boolean
}
```


### 4. RAG Engine

**Technology:** MongoDB Atlas Vector Search, Google Gemini API

**Responsibilities:**
- Generate contextually relevant interview questions
- Create vector embeddings for questions and answers
- Perform similarity search for adaptive questioning
- Evaluate answer quality and assign scores
- Generate follow-up questions for weak answers
- Maintain session context

**Core Functions:**

```python
class RAGEngine:
    def __init__(self, gemini_api_key: str, mongodb_uri: str):
        self.gemini_client = GeminiClient(api_key)
        self.vector_store = MongoDBVectorStore(mongodb_uri)
        self.embedding_model = "gemini-embedding-001"
        self.generation_model = "gemini-2.0-flash-lite"
    
    def generate_initial_questions(
        self, 
        student_profile: StudentProfile,
        num_questions: int = 1
    ) -> List[Question]:
        """Generate initial questions based on student profile"""
        pass
    
    def create_embedding(self, text: str) -> List[float]:
        """Create vector embedding using Gemini"""
        pass
    
    def search_similar_questions(
        self,
        query_embedding: List[float],
        filters: Dict,
        limit: int = 5
    ) -> List[Question]:
        """Search for similar questions using vector similarity"""
        pass
    
    def evaluate_answer(
        self,
        question: Question,
        answer: str,
        context: SessionContext
    ) -> AnswerEvaluation:
        """Evaluate answer quality and assign score (0-5)"""
        pass
    
    def generate_followup_question(
        self,
        original_question: Question,
        weak_answer: str,
        identified_gaps: List[str]
    ) -> Question:
        """Generate targeted follow-up question"""
        pass
    
    def generate_next_question(
        self,
        session_context: SessionContext,
        previous_topics: List[str]
    ) -> Question:
        """Generate new question on different topic"""
        pass
```

**Vector Search Configuration:**

```javascript
// MongoDB Atlas Vector Search Index
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 768,  // Gemini embedding dimension
        "similarity": "cosine"
      },
      "category": {
        "type": "string"
      },
      "difficulty": {
        "type": "string"
      },
      "tags": {
        "type": "string"
      }
    }
  }
}
```


### 5. Speech Emotion Analyzer

**Technology:** Python, librosa, TensorFlow, wav2vec 2.0

**Responsibilities:**
- Extract audio features (pitch, tone, pace, energy)
- Detect emotional states from speech
- Calculate confidence scores
- Identify speech patterns (hesitation, filler words)

**Core Functions:**

```python
class SpeechEmotionAnalyzer:
    def __init__(self, model_path: str):
        self.wav2vec_model = load_wav2vec_model(model_path)
        self.feature_extractor = AudioFeatureExtractor()
    
    def analyze_audio(self, audio_data: bytes) -> SpeechAnalysis:
        """Analyze audio and return comprehensive speech metrics"""
        pass
    
    def extract_features(self, audio_data: bytes) -> AudioFeatures:
        """Extract acoustic features using librosa"""
        # Pitch (F0)
        # Spectral features (MFCC, spectral centroid)
        # Energy/amplitude
        # Zero-crossing rate
        # Tempo/pace
        pass
    
    def detect_emotion(self, audio_features: AudioFeatures) -> EmotionPrediction:
        """Detect emotion using wav2vec 2.0"""
        pass
    
    def calculate_confidence_score(self, analysis: SpeechAnalysis) -> float:
        """Calculate confidence score from speech characteristics"""
        # Factors: pitch stability, pace consistency, energy level
        pass
    
    def detect_speech_pace(self, audio_data: bytes, transcript: str) -> PaceMetrics:
        """Calculate words per minute and pace classification"""
        pass
    
    def detect_filler_words(self, transcript: str) -> FillerWordAnalysis:
        """Identify filler words (um, uh, like, you know)"""
        pass

# Data Structures
class AudioFeatures:
    pitch_mean: float
    pitch_std: float
    mfcc: np.ndarray
    spectral_centroid: float
    energy: float
    zero_crossing_rate: float
    tempo: float

class SpeechAnalysis:
    emotion: str  # confident, nervous, hesitant, enthusiastic
    emotion_confidence: float
    confidence_score: float  # 0-100
    pace_wpm: float
    pace_category: str  # too_fast, optimal, too_slow
    filler_word_count: int
    pitch_variation: float
    energy_level: str  # low, medium, high
```


### 6. Facial Emotion Analyzer

**Technology:** Python, OpenCV, DeepFace, MediaPipe

**Responsibilities:**
- Detect faces in video frames
- Analyze facial expressions and emotions
- Track emotion changes over time
- Detect attention issues (face not visible)

**Core Functions:**

```python
class FacialEmotionAnalyzer:
    def __init__(self):
        self.face_detector = cv2.CascadeClassifier()
        self.deepface_model = DeepFace
        self.mediapipe_face = mp.solutions.face_mesh
        self.min_fps = 5  # Process at least 5 frames per second
    
    def analyze_frame(self, frame: np.ndarray) -> FacialAnalysis:
        """Analyze single video frame for facial emotions"""
        pass
    
    def detect_face(self, frame: np.ndarray) -> Optional[FaceRegion]:
        """Detect face in frame using OpenCV"""
        pass
    
    def analyze_emotion(self, face_region: np.ndarray) -> EmotionPrediction:
        """Analyze facial emotion using DeepFace"""
        # Returns: happy, sad, nervous, confident, neutral, surprised
        pass
    
    def extract_landmarks(self, frame: np.ndarray) -> FacialLandmarks:
        """Extract facial landmarks using MediaPipe"""
        pass
    
    def track_emotion_over_time(
        self, 
        emotion_history: List[EmotionPrediction]
    ) -> EmotionTimeline:
        """Track emotion changes throughout session"""
        pass
    
    def detect_attention_issues(
        self,
        frame_history: List[Optional[FaceRegion]],
        time_window: float = 5.0
    ) -> bool:
        """Detect if face not visible for extended period"""
        pass

# Data Structures
class FacialAnalysis:
    emotion: str
    emotion_probabilities: Dict[str, float]
    face_detected: bool
    landmarks: Optional[FacialLandmarks]
    timestamp: float

class EmotionTimeline:
    dominant_emotion: str
    emotion_distribution: Dict[str, float]
    emotion_changes: List[EmotionChange]
    stability_score: float  # How stable emotions are
```


### 7. Eye Tracker

**Technology:** Python, MediaPipe Face Mesh, OpenCV

**Responsibilities:**
- Track eye gaze direction
- Calculate attention scores
- Detect eye contact with camera
- Identify problematic gaze patterns

**Core Functions:**

```python
class EyeTracker:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.eye_landmarks = [33, 133, 160, 159, 158, 144, 145, 153]  # Key eye points
    
    def analyze_gaze(self, frame: np.ndarray) -> GazeAnalysis:
        """Analyze gaze direction from video frame"""
        pass
    
    def extract_eye_landmarks(self, frame: np.ndarray) -> Optional[EyeLandmarks]:
        """Extract eye-specific landmarks from face mesh"""
        pass
    
    def calculate_gaze_direction(self, eye_landmarks: EyeLandmarks) -> GazeVector:
        """Calculate 3D gaze direction vector"""
        pass
    
    def is_looking_at_camera(self, gaze_vector: GazeVector) -> bool:
        """Determine if user is looking at camera (eye contact)"""
        # Threshold-based detection
        pass
    
    def calculate_attention_score(
        self,
        gaze_history: List[GazeAnalysis],
        time_window: float = 10.0
    ) -> float:
        """Calculate attention score (0-100) based on gaze patterns"""
        # Factors: eye contact percentage, gaze stability
        pass
    
    def detect_gaze_patterns(
        self,
        gaze_history: List[GazeAnalysis]
    ) -> GazePatterns:
        """Identify patterns like frequent looking down/away"""
        pass

# Data Structures
class GazeAnalysis:
    gaze_vector: Tuple[float, float, float]  # 3D direction
    looking_at_camera: bool
    confidence: float
    timestamp: float

class GazePatterns:
    eye_contact_percentage: float
    avg_look_away_duration: float
    frequent_direction: str  # down, left, right, up
    attention_score: float
```


### 8. Multimodal Fusion Engine

**Technology:** Python, NumPy, Pandas, Scikit-learn

**Responsibilities:**
- Combine signals from all analyzers
- Apply weighted fusion algorithms
- Generate comprehensive performance scores
- Detect cross-modal patterns and contradictions
- Produce actionable insights

**Core Functions:**

```python
class MultimodalFusionEngine:
    def __init__(self, weights: FusionWeights = None):
        # Default weights: speech=0.4, facial=0.3, eye=0.3
        self.weights = weights or FusionWeights(0.4, 0.3, 0.3)
        self.temporal_aligner = TemporalAligner()
    
    def fuse_multimodal_data(
        self,
        speech_analysis: SpeechAnalysis,
        facial_analysis: FacialAnalysis,
        gaze_analysis: GazeAnalysis
    ) -> MultimodalScore:
        """Combine all modality signals into unified score"""
        pass
    
    def align_temporal_streams(
        self,
        speech_stream: List[SpeechAnalysis],
        facial_stream: List[FacialAnalysis],
        gaze_stream: List[GazeAnalysis]
    ) -> AlignedStreams:
        """Align timestamps across different modality streams"""
        pass
    
    def calculate_overall_score(
        self,
        aligned_data: AlignedStreams
    ) -> float:
        """Calculate weighted overall performance score"""
        pass
    
    def detect_contradictions(
        self,
        multimodal_data: AlignedStreams
    ) -> List[Contradiction]:
        """Identify contradictions between modalities"""
        # Example: confident speech but nervous facial expression
        pass
    
    def generate_insights(
        self,
        session_data: SessionData
    ) -> List[Insight]:
        """Generate actionable insights from patterns"""
        pass
    
    def calculate_confidence_intervals(
        self,
        scores: List[float]
    ) -> ConfidenceInterval:
        """Calculate statistical confidence in scores"""
        pass

# Data Structures
class FusionWeights:
    speech: float
    facial: float
    eye: float

class MultimodalScore:
    overall_score: float  # 0-100
    confidence_score: float
    attention_score: float
    emotional_stability: float
    component_scores: Dict[str, float]
    contradictions: List[Contradiction]

class Contradiction:
    modalities: Tuple[str, str]
    description: str
    severity: str  # low, medium, high
    timestamp: float

class Insight:
    category: str  # strength, weakness, pattern
    description: str
    evidence: List[str]
    recommendation: str
```


### 9. Session Orchestrator

**Technology:** Node.js/TypeScript

**Responsibilities:**
- Coordinate all components during interview session
- Manage session lifecycle
- Route data between components
- Handle component failures gracefully
- Maintain session state

**Core Functions:**

```typescript
class SessionOrchestrator {
  private ragEngine: RAGEngineClient
  private analysisService: AnalysisServiceClient
  private fusionEngine: FusionEngineClient
  private storage: StorageService
  
  async startSession(config: SessionConfig): Promise<Session> {
    // Create session record
    // Initialize all components
    // Generate first question
  }
  
  async processAnswer(
    sessionId: string,
    audioData: Buffer,
    videoFrames: ImageData[],
    transcript: string
  ): Promise<AnswerResult> {
    // Send data to analysis pipeline (parallel)
    // Wait for all analysis results
    // Fuse multimodal data
    // Evaluate answer with RAG
    // Generate next question
    // Store results
    // Return feedback
  }
  
  async endSession(sessionId: string): Promise<SessionSummary> {
    // Finalize analysis
    // Generate report
    // Store media to S3
    // Update session record
  }
  
  private async handleComponentFailure(
    component: string,
    error: Error
  ): Promise<void> {
    // Log error
    // Continue with available components
    // Notify user of degraded functionality
  }
}
```


### 10. Report Generator

**Technology:** Node.js, PDFKit or similar

**Responsibilities:**
- Generate comprehensive PDF reports
- Include charts and visualizations
- Format transcript with timestamps
- Provide actionable recommendations

**Report Structure:**

```typescript
interface SessionReport {
  metadata: {
    sessionId: string
    studentName: string
    date: Date
    duration: number
    questionsAsked: number
  }
  
  overallSummary: {
    overallScore: number
    confidenceScore: number
    attentionScore: number
    emotionalStability: number
    strengths: string[]
    weaknesses: string[]
  }
  
  questionAnalysis: Array<{
    question: string
    answer: string
    score: number
    feedback: string
    timestamp: number
  }>
  
  multimodalAnalysis: {
    speechMetrics: {
      avgPace: number
      pitchVariation: number
      fillerWordCount: number
      emotionDistribution: Record<string, number>
    }
    facialMetrics: {
      emotionDistribution: Record<string, number>
      emotionStability: number
    }
    eyeTrackingMetrics: {
      eyeContactPercentage: number
      avgLookAwayDuration: number
      attentionScore: number
    }
  }
  
  visualizations: {
    emotionTimeline: ChartData
    scoreProgression: ChartData
    modalityComparison: ChartData
  }
  
  recommendations: Array<{
    category: string
    priority: 'high' | 'medium' | 'low'
    recommendation: string
    actionableSteps: string[]
  }>
  
  transcript: Array<{
    timestamp: number
    speaker: 'interviewer' | 'student'
    text: string
  }>
}
```


## Data Models

### MongoDB Collections

**users**
```javascript
{
  _id: ObjectId,
  email: string,
  passwordHash: string,
  role: 'student' | 'employer',
  profile: {
    name: string,
    university: string,
    targetRole: string,
    graduationYear: number
  },
  createdAt: Date,
  updatedAt: Date
}
```

**sessions**
```javascript
{
  _id: ObjectId,
  userId: ObjectId,
  status: 'active' | 'completed' | 'abandoned',
  startTime: Date,
  endTime: Date,
  duration: number,  // seconds
  config: {
    interviewType: string,
    difficulty: string
  },
  questions: [{
    questionId: string,
    questionText: string,
    askedAt: Date,
    answer: {
      text: string,
      audioUrl: string,
      score: number,  // 0-5
      feedback: string
    },
    isFollowUp: boolean
  }],
  overallMetrics: {
    overallScore: number,
    confidenceScore: number,
    attentionScore: number,
    emotionalStability: number
  },
  multimodalData: {
    speechAnalysis: Object,
    facialAnalysis: Object,
    gazeAnalysis: Object
  },
  reportUrl: string,
  createdAt: Date,
  updatedAt: Date
}
```

**questions**
```javascript
{
  _id: ObjectId,
  text: string,
  category: string,  // technical, behavioral, situational
  difficulty: string,  // easy, medium, hard
  tags: [string],
  embedding: [float],  // 768-dimensional vector
  expectedKeywords: [string],
  sampleAnswer: string,
  createdAt: Date
}
```


### Redis Cache Structure

**Authentication Tokens**
```
Key: auth:token:{userId}
Value: JWT token string
TTL: Token expiration time
```

**Session State**
```
Key: session:state:{sessionId}
Value: JSON serialized session state
TTL: 1 hour
```

**Question Embeddings Cache**
```
Key: embedding:question:{questionId}
Value: JSON array of embedding vector
TTL: 7 days
```

**User Analytics Cache**
```
Key: analytics:user:{userId}
Value: JSON serialized analytics data
TTL: 1 hour
```

**Employer Dashboard Cache**
```
Key: analytics:employer:dashboard
Value: JSON serialized aggregated metrics
TTL: 15 minutes
```

### AWS S3 Structure

**Audio Recordings**
```
Path: audio/{userId}/{sessionId}/{timestamp}.webm
Metadata: {
  contentType: 'audio/webm',
  userId: string,
  sessionId: string,
  duration: number
}
Encryption: AES-256
```

**Video Recordings**
```
Path: video/{userId}/{sessionId}/{timestamp}.webm
Metadata: {
  contentType: 'video/webm',
  userId: string,
  sessionId: string,
  duration: number,
  resolution: string
}
Encryption: AES-256
```

**Session Reports**
```
Path: reports/{userId}/{sessionId}/report.pdf
Metadata: {
  contentType: 'application/pdf',
  userId: string,
  sessionId: string,
  generatedAt: Date
}
Encryption: AES-256
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Authentication and Authorization Properties

**Property 1: Valid credentials produce valid JWT tokens**
*For any* valid user credentials, authentication should return a valid JWT token that can be used for subsequent requests.
**Validates: Requirements 1.1, 1.3**

**Property 2: Invalid credentials are rejected**
*For any* invalid credential combination (wrong password, non-existent user, malformed input), authentication should be rejected with an appropriate error message.
**Validates: Requirements 1.2**

**Property 3: Role-based access control**
*For any* authenticated user, the accessible endpoints should match their role permissions (students access practice/analytics, employers access employer dashboard).
**Validates: Requirements 1.5, 1.6**

**Property 4: Password hashing**
*For any* user account, the stored password should be a cryptographic hash, not plaintext.
**Validates: Requirements 1.7**


### RAG Engine Properties

**Property 5: Question generation for all profiles**
*For any* student profile, the RAG engine should generate at least one relevant initial question.
**Validates: Requirements 2.1**

**Property 6: Embedding dimension consistency**
*For any* text input, generated embeddings should have exactly 768 dimensions (Gemini embedding-001 output size).
**Validates: Requirements 2.2**

**Property 7: Vector search ordering**
*For any* query embedding, search results should be ordered by descending cosine similarity scores.
**Validates: Requirements 2.3**

**Property 8: Score bounds**
*For any* answer evaluation, the assigned Question_Score should be between 0 and 5 inclusive.
**Validates: Requirements 2.4**

**Property 9: Adaptive questioning based on score**
*For any* answer with score < 3, the next question should be a follow-up targeting identified gaps; for any answer with score >= 3, the next question should be on a different topic.
**Validates: Requirements 2.5, 2.6**

**Property 10: Question uniqueness within session**
*For any* session, no question should be asked more than once.
**Validates: Requirements 2.8**

**Property 11: Feedback presence**
*For any* evaluated answer, feedback should be non-empty and contain specific strengths or improvement areas.
**Validates: Requirements 2.9**


### Speech Analysis Properties

**Property 12: Audio feature extraction completeness**
*For any* valid audio input, the Speech_Analyzer should extract all required features: pitch, tone, pace, and energy metrics.
**Validates: Requirements 3.2**

**Property 13: Speech emotion classification validity**
*For any* audio analysis, the detected emotion should be from the valid set: {confident, nervous, hesitant, enthusiastic}.
**Validates: Requirements 3.4**

**Property 14: Speech pace calculation**
*For any* audio with transcript, the calculated words-per-minute should match the actual word count divided by duration.
**Validates: Requirements 3.5**

**Property 15: Pitch variation detection**
*For any* audio input, pitch analysis should detect and quantify variations or identify monotone patterns.
**Validates: Requirements 3.6**

**Property 16: Confidence score generation**
*For any* speech analysis, a Confidence_Score between 0 and 100 should be generated based on speech characteristics.
**Validates: Requirements 3.7**

### Facial Analysis Properties

**Property 17: Facial emotion classification validity**
*For any* frame with detected face, the classified emotion should be from the valid set: {happy, sad, nervous, confident, neutral, surprised}.
**Validates: Requirements 4.4**

**Property 18: Emotion tracking continuity**
*For any* sequence of video frames, emotion changes should be tracked with timestamps throughout the session.
**Validates: Requirements 4.6**


### Eye Tracking Properties

**Property 19: Gaze direction calculation**
*For any* facial landmarks containing eye positions, a gaze direction vector should be calculated.
**Validates: Requirements 5.2**

**Property 20: Eye contact detection**
*For any* gaze direction vector, the system should determine whether the user is looking at the camera.
**Validates: Requirements 5.3**

**Property 21: Attention score bounds**
*For any* gaze pattern analysis, the computed Attention_Score should be between 0 and 100.
**Validates: Requirements 5.4**

**Property 22: Gaze pattern identification**
*For any* sequence of gaze data, patterns such as frequent looking down or sideways should be identified and quantified.
**Validates: Requirements 5.6**

**Property 23: Session-wide attention tracking**
*For any* active session, attention metrics should be tracked continuously from start to end.
**Validates: Requirements 5.8**

### Multimodal Fusion Properties

**Property 24: Multimodal data combination**
*For any* set of speech, facial, and eye tracking data, the Fusion_Engine should combine all available modalities.
**Validates: Requirements 6.1**

**Property 25: Weighted fusion**
*For any* multimodal data with different weight configurations, the overall score should change proportionally to the weights.
**Validates: Requirements 6.2**

**Property 26: Overall score generation**
*For any* multimodal analysis, an overall performance score should be generated combining all modalities.
**Validates: Requirements 6.3**

**Property 27: Contradiction detection**
*For any* multimodal data where signals conflict (e.g., confident speech with nervous facial expression), contradictions should be flagged.
**Validates: Requirements 6.4**

**Property 28: Insight generation**
*For any* multimodal session data, actionable insights should be generated based on identified patterns.
**Validates: Requirements 6.7**

**Property 29: Temporal alignment**
*For any* multimodal streams with timestamps, data from different modalities should be aligned temporally before fusion.
**Validates: Requirements 6.8**


### Real-Time Feedback Properties

**Property 30: Real-time metrics delivery**
*For any* active session, real-time metrics (Confidence_Score, Attention_Score, emotion states) should be sent to the frontend.
**Validates: Requirements 7.2**

**Property 31: Pace indicator provision**
*For any* speech analysis, pace indicators (too_fast, optimal, too_slow) should be included in feedback.
**Validates: Requirements 7.7**

**Property 32: Cumulative statistics calculation**
*For any* active session, cumulative statistics including average scores should be calculated and updated.
**Validates: Requirements 7.8**

### Session Management Properties

**Property 33: Session record creation**
*For any* started session, a new Session record should be created in MongoDB with a unique session ID.
**Validates: Requirements 8.1**

**Property 34: Session data persistence**
*For any* active session, all questions, answers, and analysis data should be stored as they occur.
**Validates: Requirements 8.2**

**Property 35: Session completion persistence**
*For any* ended session, the complete transcript and all metrics should be saved to MongoDB.
**Validates: Requirements 8.3**

**Property 36: Media storage with access controls**
*For any* audio or video recording, the file should be stored in AWS S3 with server-side encryption and appropriate access controls.
**Validates: Requirements 8.4, 8.5**

**Property 37: Session history retrieval and sorting**
*For any* student requesting session history, all past sessions should be retrieved and sorted by date (most recent first).
**Validates: Requirements 8.6**

**Property 38: Session summary completeness**
*For any* session summary, it should include date, duration, overall score, and key metrics.
**Validates: Requirements 8.7**

**Property 39: Session detail retrieval**
*For any* selected past session, complete details including transcript and analysis should be retrievable.
**Validates: Requirements 8.8**

**Property 40: Session comparison**
*For any* set of multiple sessions, metrics should be comparable across sessions.
**Validates: Requirements 8.9**


### Report Generation Properties

**Property 41: PDF report generation**
*For any* completed session, a PDF report should be generated.
**Validates: Requirements 9.1**

**Property 42: Report content completeness**
*For any* generated report, it should include all required sections: metadata, transcript with timestamps, question scores with feedback, emotion charts, speech metrics, attention metrics, performance summary, and actionable recommendations.
**Validates: Requirements 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9**

**Property 43: Report storage and access**
*For any* generated report, it should be stored in AWS S3 and a download link should be provided.
**Validates: Requirements 9.10**

### Employer Analytics Properties

**Property 44: Employer dashboard access**
*For any* user with employer role, the employer analytics dashboard should be accessible.
**Validates: Requirements 10.1**

**Property 45: Metrics aggregation**
*For any* employer analytics request, metrics should be aggregated correctly across all students.
**Validates: Requirements 10.2**

**Property 46: Score distribution calculation**
*For any* set of sessions, performance score distributions across interview categories should be calculated correctly.
**Validates: Requirements 10.3**

**Property 47: Performance trend calculation**
*For any* time-series session data, performance trends over time should be computed.
**Validates: Requirements 10.4**

**Property 48: Analytics filtering**
*For any* filter criteria (date range, interview type, performance level), analytics results should include only sessions matching the criteria.
**Validates: Requirements 10.5**

**Property 49: Top performer identification**
*For any* set of students with sessions, top performers should be ranked correctly by overall scores.
**Validates: Requirements 10.6**

**Property 50: Common weakness identification**
*For any* set of sessions, common weaknesses appearing across multiple students should be identified and ranked by frequency.
**Validates: Requirements 10.7**

**Property 51: Privacy protection in aggregated data**
*For any* employer analytics data, personally identifiable information should not be exposed without explicit consent.
**Validates: Requirements 10.8**


### Caching Properties

**Property 52: Cache TTL configuration**
*For any* cached data, appropriate TTL values should be set (authentication tokens match JWT expiry, session data has reasonable expiration).
**Validates: Requirements 11.3, 11.4**

**Property 53: Cache refresh on expiration**
*For any* expired cache entry that is requested, data should be refreshed from primary storage.
**Validates: Requirements 11.5**

**Property 54: Cache invalidation on data change**
*For any* data modification, related cache entries should be invalidated.
**Validates: Requirements 11.6**

### Media Processing Properties

**Property 55: Media compression**
*For any* media file uploaded to S3, it should be compressed before upload.
**Validates: Requirements 12.3**

**Property 56: S3 encryption**
*For any* media file stored in S3, server-side encryption should be enabled.
**Validates: Requirements 12.4**

**Property 57: Pre-signed URL generation**
*For any* media access request, a pre-signed URL with time-limited access should be generated.
**Validates: Requirements 12.5**

**Property 58: Upload retry logic**
*For any* failed media upload, retry attempts should be made before reporting failure.
**Validates: Requirements 12.7**

**Property 59: Media retention policy**
*For any* media file older than the retention period (default 90 days), it should be deleted from S3.
**Validates: Requirements 12.8**


### System Resilience Properties

**Property 60: Graceful component degradation**
*For any* analyzer component failure (Speech, Facial, or Eye Tracker), the system should continue operating with remaining functional components.
**Validates: Requirements 13.1, 13.2, 13.3**

**Property 61: RAG fallback**
*For any* RAG_Engine error, the system should fall back to pre-defined question sets.
**Validates: Requirements 13.4**

**Property 62: Error logging**
*For any* error that occurs, it should be logged with sufficient context (timestamp, component, error details, user context).
**Validates: Requirements 13.8**

**Property 63: Critical error notification**
*For any* critical error, administrators should be notified via configured channels.
**Validates: Requirements 13.9**

### Frontend Properties

**Property 64: Loading indicator display**
*For any* processing operation, loading indicators should be displayed to the user.
**Validates: Requirements 14.8**

**Property 65: Error message display**
*For any* error that affects the user, a user-friendly error message with suggested actions should be displayed.
**Validates: Requirements 14.9**

### API Properties

**Property 66: Input validation with status codes**
*For any* API request, inputs should be validated and appropriate HTTP status codes returned (400 for validation errors, 401 for auth failures, 403 for authorization failures).
**Validates: Requirements 15.4**

**Property 67: Rate limiting enforcement**
*For any* client making excessive requests, rate limiting should be enforced to prevent abuse.
**Validates: Requirements 15.8**


## Error Handling

### Error Categories

**1. Authentication/Authorization Errors**
- Invalid credentials → 401 Unauthorized with message "Invalid email or password"
- Expired token → 401 Unauthorized with message "Session expired, please login again"
- Insufficient permissions → 403 Forbidden with message "Access denied"
- Malformed token → 401 Unauthorized with message "Invalid authentication token"

**2. Validation Errors**
- Missing required fields → 400 Bad Request with field-specific messages
- Invalid data types → 400 Bad Request with type mismatch details
- Out-of-range values → 400 Bad Request with valid range information
- Invalid format → 400 Bad Request with format requirements

**3. Resource Errors**
- Session not found → 404 Not Found with message "Session does not exist"
- User not found → 404 Not Found with message "User not found"
- Report not found → 404 Not Found with message "Report not available"
- Media not found → 404 Not Found with message "Media file not found"

**4. Service Errors**
- RAG Engine failure → Fallback to pre-defined questions, log error, continue session
- Speech Analyzer failure → Continue with facial and eye tracking, mark speech data unavailable
- Facial Analyzer failure → Continue with speech and eye tracking, mark facial data unavailable
- Eye Tracker failure → Continue with speech and facial analysis, mark eye data unavailable
- MongoDB unavailable → Queue writes, retry with exponential backoff (1s, 2s, 4s, 8s, 16s max)
- Redis unavailable → Fall back to direct database queries, log warning
- S3 unavailable → Store media temporarily in local storage, retry upload, notify user of delay

**5. External API Errors**
- Gemini API timeout → Retry up to 3 times with exponential backoff
- Gemini API rate limit → Queue request, retry after rate limit window
- Gemini API error → Log error, return cached results if available, or generic fallback

**6. Media Processing Errors**
- Audio codec unsupported → Return 400 with supported codec list
- Video codec unsupported → Return 400 with supported codec list
- File too large → Return 413 Payload Too Large with size limit
- Corrupted media → Return 400 with message "Media file corrupted"
- Upload failure → Retry up to 3 times, then return 500 with retry suggestion

### Error Response Format

All API errors follow consistent JSON structure:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message",
    "details": {
      "field": "specific field if applicable",
      "reason": "detailed reason",
      "suggestion": "suggested action"
    },
    "timestamp": "2024-01-15T10:30:00Z",
    "requestId": "unique-request-id"
  }
}
```

### Graceful Degradation Strategy

The system prioritizes continuity over perfection:

1. **Component Isolation**: Each analyzer runs independently; failure of one doesn't crash others
2. **Partial Results**: Return available analysis even if some components fail
3. **User Notification**: Inform users when functionality is degraded
4. **Automatic Recovery**: Retry failed operations automatically when possible
5. **Fallback Mechanisms**: Use simpler alternatives when primary methods fail


## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across a wide input space.

### Property-Based Testing

**Framework Selection:**
- **JavaScript/TypeScript**: fast-check
- **Python**: Hypothesis

**Configuration:**
- Minimum 100 iterations per property test (due to randomization)
- Each property test must reference its design document property
- Tag format: `Feature: ai-multimodal-interview-assistant, Property {number}: {property_text}`

**Property Test Examples:**

```typescript
// Property 1: Valid credentials produce valid JWT tokens
import fc from 'fast-check'

test('Property 1: Valid credentials produce valid JWT tokens', async () => {
  await fc.assert(
    fc.asyncProperty(
      fc.record({
        email: fc.emailAddress(),
        password: fc.string({ minLength: 8 })
      }),
      async (credentials) => {
        // Feature: ai-multimodal-interview-assistant, Property 1
        const user = await createUser(credentials)
        const result = await authenticate(credentials)
        
        expect(result.token).toBeDefined()
        expect(verifyJWT(result.token)).toBeTruthy()
        expect(decodeJWT(result.token).userId).toBe(user.id)
      }
    ),
    { numRuns: 100 }
  )
})
```

```python
# Property 12: Audio feature extraction completeness
from hypothesis import given, strategies as st
import numpy as np

@given(st.binary(min_size=1000, max_size=100000))
def test_audio_feature_extraction_completeness(audio_data):
    """Feature: ai-multimodal-interview-assistant, Property 12"""
    analyzer = SpeechEmotionAnalyzer()
    features = analyzer.extract_features(audio_data)
    
    assert features.pitch_mean is not None
    assert features.mfcc is not None
    assert features.spectral_centroid is not None
    assert features.energy is not None
```


### Unit Testing Strategy

**Focus Areas for Unit Tests:**

1. **Specific Examples**
   - Test authentication with known credentials
   - Test specific question-answer pairs
   - Test known emotion patterns

2. **Edge Cases**
   - Expired JWT tokens (Requirement 1.4)
   - No face detected for > 5 seconds (Requirement 4.7)
   - Looking away for > 3 seconds (Requirement 5.5)
   - Poor audio quality (Requirement 3.8)
   - High communication latency (Requirement 7.4)
   - Large media files > 100MB (Requirement 12.6)
   - MongoDB unavailable (Requirement 13.5)
   - S3 unavailable (Requirement 13.6)
   - API timeouts (Requirement 13.7)
   - Cache unavailable (Requirement 11.7)

3. **Error Conditions**
   - Invalid API inputs
   - Component failures
   - Network errors
   - Database errors

4. **Integration Points**
   - API endpoint contracts
   - WebSocket event handling
   - Database operations
   - External API calls

**Unit Test Examples:**

```typescript
// Edge case: Expired JWT token
test('Expired JWT token requires re-authentication', async () => {
  const user = await createUser({ email: 'test@example.com', password: 'password123' })
  const token = generateExpiredToken(user.id)
  
  const response = await request(app)
    .get('/api/sessions')
    .set('Authorization', `Bearer ${token}`)
  
  expect(response.status).toBe(401)
  expect(response.body.error.message).toContain('expired')
})

// Example: Specific HTTP status codes
test('Validation failure returns 400 Bad Request', async () => {
  const response = await request(app)
    .post('/api/sessions')
    .send({ /* missing required fields */ })
  
  expect(response.status).toBe(400)
  expect(response.body.error.details).toBeDefined()
})

test('Authentication failure returns 401 Unauthorized', async () => {
  const response = await request(app)
    .post('/api/auth/login')
    .send({ email: 'test@example.com', password: 'wrongpassword' })
  
  expect(response.status).toBe(401)
})

test('Authorization failure returns 403 Forbidden', async () => {
  const studentToken = await getStudentToken()
  const response = await request(app)
    .get('/api/employer/dashboard')
    .set('Authorization', `Bearer ${studentToken}`)
  
  expect(response.status).toBe(403)
})
```

```python
# Edge case: No face detected for extended period
def test_attention_issue_flagged_when_no_face_for_5_seconds():
    analyzer = FacialEmotionAnalyzer()
    frames = [None] * 150  # 30 fps * 5 seconds = 150 frames
    
    attention_issue = analyzer.detect_attention_issues(frames, time_window=5.0)
    
    assert attention_issue is True

# Edge case: Large file multipart upload
def test_multipart_upload_for_large_files():
    large_file = generate_media_file(size_mb=150)
    
    uploader = MediaUploader()
    result = uploader.upload(large_file)
    
    assert result.used_multipart is True
    assert result.success is True
```


### Integration Testing

**Key Integration Points:**

1. **Frontend ↔ API Server**
   - Authentication flow
   - Session CRUD operations
   - Report downloads

2. **Frontend ↔ Real-Time Server**
   - WebSocket connection establishment
   - Real-time event streaming
   - Session state synchronization

3. **API Server ↔ MongoDB**
   - User management
   - Session persistence
   - Query operations

4. **API Server ↔ Redis**
   - Cache operations
   - Token storage
   - Analytics caching

5. **Backend ↔ AWS S3**
   - Media upload
   - Pre-signed URL generation
   - File deletion

6. **Backend ↔ Google Gemini API**
   - Embedding generation
   - Question generation
   - Answer evaluation

7. **Analysis Pipeline Integration**
   - Speech → Facial → Eye → Fusion
   - Parallel processing coordination
   - Result aggregation

**Integration Test Example:**

```typescript
// End-to-end session flow
test('Complete interview session flow', async () => {
  // 1. Authenticate
  const authResponse = await authenticate({ email: 'student@test.com', password: 'pass123' })
  const token = authResponse.token
  
  // 2. Start session
  const sessionResponse = await startSession(token, { interviewType: 'technical' })
  const sessionId = sessionResponse.sessionId
  
  // 3. Connect WebSocket
  const socket = await connectWebSocket(token)
  
  // 4. Receive first question
  const question = await waitForEvent(socket, 'question:new')
  expect(question.text).toBeDefined()
  
  // 5. Submit answer
  await submitAnswer(socket, { audio: mockAudio, text: 'My answer' })
  
  // 6. Receive feedback
  const feedback = await waitForEvent(socket, 'feedback:answer')
  expect(feedback.score).toBeGreaterThanOrEqual(0)
  expect(feedback.score).toBeLessThanOrEqual(5)
  
  // 7. End session
  await endSession(socket)
  
  // 8. Verify session saved
  const savedSession = await getSession(token, sessionId)
  expect(savedSession.status).toBe('completed')
  expect(savedSession.questions.length).toBeGreaterThan(0)
  
  // 9. Verify report generated
  expect(savedSession.reportUrl).toBeDefined()
})
```

### Test Coverage Goals

- **Unit Test Coverage**: Minimum 80% code coverage
- **Property Test Coverage**: All 67 correctness properties implemented
- **Integration Test Coverage**: All major user flows covered
- **Edge Case Coverage**: All identified edge cases tested

### Continuous Testing

- Run unit tests on every commit
- Run property tests on every pull request
- Run integration tests before deployment
- Monitor test execution time and optimize slow tests
- Maintain test data generators for consistent test scenarios

