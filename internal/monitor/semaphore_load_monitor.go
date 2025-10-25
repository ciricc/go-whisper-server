package monitor

import (
	"sync/atomic"

	"golang.org/x/sync/semaphore"
)

// SemaphoreLoadMonitor implements LoadMonitor using a semaphore for concurrency control.
// It tracks active tasks by monitoring semaphore acquisition/release.
type SemaphoreLoadMonitor struct {
	sem       *semaphore.Weighted
	maxWeight int64
	activeCnt atomic.Int64
	threshold float64 // 0.0 - 1.0, percentage of max capacity
}

// NewSemaphoreLoadMonitor creates a new semaphore-based load monitor.
// maxConcurrency is the maximum number of concurrent tasks allowed.
// healthThreshold is the load percentage (0.0-1.0) above which the service is considered unhealthy.
// For example, 0.8 means the service is unhealthy when >80% of capacity is used.
func NewSemaphoreLoadMonitor(maxConcurrency int64, healthThreshold float64) *SemaphoreLoadMonitor {
	if healthThreshold < 0.0 {
		healthThreshold = 0.0
	}
	if healthThreshold > 1.0 {
		healthThreshold = 1.0
	}

	return &SemaphoreLoadMonitor{
		sem:       semaphore.NewWeighted(maxConcurrency),
		maxWeight: maxConcurrency,
		threshold: healthThreshold,
	}
}

// GetMetrics returns current load statistics
func (m *SemaphoreLoadMonitor) GetMetrics() LoadMetrics {
	active := m.activeCnt.Load()
	loadPct := 0.0
	if m.maxWeight > 0 {
		loadPct = float64(active) / float64(m.maxWeight) * 100.0
	}

	return LoadMetrics{
		ActiveTasks:    active,
		MaxTasks:       m.maxWeight,
		LoadPercentage: loadPct,
	}
}

// CanAcceptTask returns true if the system can accept a new task.
// It uses TryAcquire to check without blocking.
func (m *SemaphoreLoadMonitor) CanAcceptTask() bool {
	if m.sem.TryAcquire(1) {
		// Immediately release since we're just checking
		m.sem.Release(1)
		return true
	}
	return false
}

// IsHealthy returns true if the current load is below the health threshold
func (m *SemaphoreLoadMonitor) IsHealthy() bool {
	metrics := m.GetMetrics()
	currentLoad := metrics.LoadPercentage / 100.0
	return currentLoad <= m.threshold
}

// TryAcquire attempts to acquire a task slot. Returns true if successful.
// The caller MUST call Release() when the task completes.
func (m *SemaphoreLoadMonitor) TryAcquire() bool {
	if m.sem.TryAcquire(1) {
		m.activeCnt.Add(1)
		return true
	}
	return false
}

// Release releases a task slot, allowing another task to be acquired
func (m *SemaphoreLoadMonitor) Release() {
	m.activeCnt.Add(-1)
	m.sem.Release(1)
}

// GetSemaphore returns the underlying semaphore (for backwards compatibility)
func (m *SemaphoreLoadMonitor) GetSemaphore() *semaphore.Weighted {
	return m.sem
}
