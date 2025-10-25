package health

import (
	"context"
	"sync"

	"github.com/ciricc/go-whisper-server/internal/monitor"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/status"
)

// HealthChecker implements the gRPC health checking protocol.
// It uses LoadMonitor to determine service health based on current load.
type HealthChecker struct {
	grpc_health_v1.UnimplementedHealthServer
	mu           sync.RWMutex
	loadMonitor  monitor.LoadMonitor
	statusMap    map[string]grpc_health_v1.HealthCheckResponse_ServingStatus
	globalStatus grpc_health_v1.HealthCheckResponse_ServingStatus
}

// NewHealthChecker creates a new health checker that monitors the given load monitor
func NewHealthChecker(loadMonitor monitor.LoadMonitor) *HealthChecker {
	return &HealthChecker{
		loadMonitor:  loadMonitor,
		statusMap:    make(map[string]grpc_health_v1.HealthCheckResponse_ServingStatus),
		globalStatus: grpc_health_v1.HealthCheckResponse_SERVING,
	}
}

// Check implements the health check RPC
func (h *HealthChecker) Check(ctx context.Context, req *grpc_health_v1.HealthCheckRequest) (*grpc_health_v1.HealthCheckResponse, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	service := req.GetService()

	// Determine status based on load
	var servingStatus grpc_health_v1.HealthCheckResponse_ServingStatus

	if service == "" {
		// Global health check - check overall system health
		if h.loadMonitor.IsHealthy() {
			servingStatus = grpc_health_v1.HealthCheckResponse_SERVING
		} else {
			servingStatus = grpc_health_v1.HealthCheckResponse_NOT_SERVING
		}
	} else {
		// Service-specific health check
		if st, ok := h.statusMap[service]; ok {
			servingStatus = st
		} else {
			return nil, status.Error(codes.NotFound, "service not found")
		}

		// Even if service status is SERVING, override if load is too high
		if servingStatus == grpc_health_v1.HealthCheckResponse_SERVING && !h.loadMonitor.IsHealthy() {
			servingStatus = grpc_health_v1.HealthCheckResponse_NOT_SERVING
		}
	}

	return &grpc_health_v1.HealthCheckResponse{
		Status: servingStatus,
	}, nil
}

// Watch implements the health check streaming RPC
func (h *HealthChecker) Watch(req *grpc_health_v1.HealthCheckRequest, stream grpc_health_v1.Health_WatchServer) error {
	service := req.GetService()

	// Send initial status
	h.mu.RLock()
	var initialStatus grpc_health_v1.HealthCheckResponse_ServingStatus
	if service == "" {
		if h.loadMonitor.IsHealthy() {
			initialStatus = grpc_health_v1.HealthCheckResponse_SERVING
		} else {
			initialStatus = grpc_health_v1.HealthCheckResponse_NOT_SERVING
		}
	} else {
		if st, ok := h.statusMap[service]; ok {
			initialStatus = st
			if initialStatus == grpc_health_v1.HealthCheckResponse_SERVING && !h.loadMonitor.IsHealthy() {
				initialStatus = grpc_health_v1.HealthCheckResponse_NOT_SERVING
			}
		} else {
			h.mu.RUnlock()
			return status.Error(codes.NotFound, "service not found")
		}
	}
	h.mu.RUnlock()

	if err := stream.Send(&grpc_health_v1.HealthCheckResponse{Status: initialStatus}); err != nil {
		return err
	}

	// Keep the stream open - in a real implementation, you'd want to send updates
	// when the status changes. For now, just block until context is cancelled.
	<-stream.Context().Done()
	return stream.Context().Err()
}

// SetServingStatus sets the serving status for a specific service
func (h *HealthChecker) SetServingStatus(service string, status grpc_health_v1.HealthCheckResponse_ServingStatus) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.statusMap[service] = status
}

// SetGlobalStatus sets the global serving status (when service name is empty)
func (h *HealthChecker) SetGlobalStatus(status grpc_health_v1.HealthCheckResponse_ServingStatus) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.globalStatus = status
}

// GetLoadMetrics returns current load metrics (useful for monitoring/debugging)
func (h *HealthChecker) GetLoadMetrics() monitor.LoadMetrics {
	return h.loadMonitor.GetMetrics()
}
