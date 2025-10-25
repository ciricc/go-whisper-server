package monitor

// LoadMetrics represents current load statistics
type LoadMetrics struct {
	// ActiveTasks is the number of currently running tasks
	ActiveTasks int64
	// MaxTasks is the maximum number of concurrent tasks allowed
	MaxTasks int64
	// LoadPercentage is the current load as a percentage (0-100)
	LoadPercentage float64
}

// LoadMonitor is an interface for monitoring system load and capacity.
// It abstracts away the implementation details of how load is tracked,
// allowing different implementations (semaphore-based, queue-based, etc.)
type LoadMonitor interface {
	// GetMetrics returns current load statistics
	GetMetrics() LoadMetrics

	// CanAcceptTask returns true if the system can accept a new task
	CanAcceptTask() bool

	// IsHealthy returns true if the system is healthy and can accept work
	// This may take into account additional factors beyond just capacity
	IsHealthy() bool

	// TryAcquire attempts to acquire a task slot. Returns true if successful.
	// The caller MUST call Release() when the task completes.
	TryAcquire() bool

	// Release releases a task slot, allowing another task to be acquired
	Release()
}
