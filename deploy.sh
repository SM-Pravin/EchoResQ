#!/bin/bash

# Emergency AI - Deployment and Management Scripts
# Comprehensive deployment automation for production and development

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DOCKER_IMAGE_NAME="emergency-ai"
DOCKER_TAG="latest"
REGISTRY_URL="your-registry.com"  # Update with your registry
PROJECT_NAME="emergency-ai"

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Emergency AI Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    build               Build Docker images
    deploy              Deploy to production
    dev                 Start development environment
    test                Run test suite
    benchmark          Run performance benchmarks
    clean              Clean up Docker resources
    logs               Show application logs
    status             Show deployment status
    backup             Backup data and models
    restore            Restore from backup
    update             Update deployment
    stop               Stop all services
    restart            Restart services
    help               Show this help message

Options:
    --env ENV          Environment (dev|staging|prod)
    --tag TAG          Docker image tag
    --no-cache         Build without cache
    --profile PROFILE  Docker compose profile
    --verbose          Verbose output

Examples:
    $0 build --tag v1.0.0
    $0 deploy --env prod
    $0 dev --profile jupyter
    $0 test --verbose
    $0 logs --env prod

EOF
}

# Check dependencies
check_dependencies() {
    log "Checking dependencies..."
    
    local deps=("docker" "docker-compose" "python3" "pip")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        error "Missing dependencies: ${missing_deps[*]}"
        error "Please install the missing dependencies and try again"
        exit 1
    fi
    
    success "All dependencies are installed"
}

# Build Docker images
build_images() {
    local tag="${1:-$DOCKER_TAG}"
    local no_cache="$2"
    
    log "Building Docker images with tag: $tag"
    
    local build_args=""
    if [ "$no_cache" = "true" ]; then
        build_args="--no-cache"
    fi
    
    # Build production image
    log "Building production image..."
    docker build $build_args -t $DOCKER_IMAGE_NAME:$tag -f Dockerfile .
    
    # Build development image
    log "Building development image..."
    docker build $build_args -t $DOCKER_IMAGE_NAME:dev -f Dockerfile.dev .
    
    success "Docker images built successfully"
}

# Deploy to production
deploy_production() {
    log "Deploying to production..."
    
    # Check if production environment is ready
    if [ ! -f "docker-compose.yml" ]; then
        error "docker-compose.yml not found"
        exit 1
    fi
    
    # Pull latest images if using registry
    # docker-compose pull
    
    # Start services
    log "Starting production services..."
    docker-compose --profile production up -d
    
    # Wait for services to be healthy
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check health
    if docker-compose ps | grep -q "unhealthy"; then
        error "Some services are unhealthy"
        docker-compose logs
        exit 1
    fi
    
    success "Production deployment completed"
    show_deployment_status
}

# Start development environment
start_development() {
    local profile="$1"
    
    log "Starting development environment..."
    
    # Use development compose file
    local compose_args=""
    if [ -n "$profile" ]; then
        compose_args="--profile $profile"
    fi
    
    docker-compose -f docker-compose.dev.yml $compose_args up -d
    
    success "Development environment started"
    log "Streamlit UI: http://localhost:8501"
    log "Jupyter Lab (if enabled): http://localhost:8888 (token: emergency123)"
}

# Run tests
run_tests() {
    local verbose="$1"
    
    log "Running test suite..."
    
    # Create test container
    docker run --rm \
        -v "$(pwd):/app" \
        -w /app \
        $DOCKER_IMAGE_NAME:dev \
        python -m pytest WORKING_FILES/tests/ \
        $([ "$verbose" = "true" ] && echo "-v" || echo "-q")
    
    success "Tests completed"
}

# Run benchmarks
run_benchmarks() {
    log "Running performance benchmarks..."
    
    docker run --rm \
        -v "$(pwd):/app" \
        -w /app \
        $DOCKER_IMAGE_NAME:dev \
        python WORKING_FILES/benchmarks/performance_profiler.py --benchmark
    
    success "Benchmarks completed"
}

# Clean up Docker resources
cleanup() {
    log "Cleaning up Docker resources..."
    
    # Stop all containers
    docker-compose down
    docker-compose -f docker-compose.dev.yml down
    
    # Remove dangling images
    docker image prune -f
    
    # Remove unused volumes (with confirmation)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    success "Cleanup completed"
}

# Show logs  
show_logs() {
    local env="${1:-dev}"
    local service="$2"
    
    if [ "$env" = "prod" ]; then
        if [ -n "$service" ]; then
            docker-compose logs -f "$service"
        else
            docker-compose logs -f
        fi
    else
        if [ -n "$service" ]; then
            docker-compose -f docker-compose.dev.yml logs -f "$service"
        else
            docker-compose -f docker-compose.dev.yml logs -f
        fi
    fi
}

# Show deployment status
show_deployment_status() {
    log "Deployment Status:"
    echo
    
    # Show running containers
    echo "Running Containers:"
    docker-compose ps
    echo
    
    # Show resource usage
    echo "Resource Usage:"
    docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"
    echo
    
    # Show endpoints
    echo "Available Endpoints:"
    echo "  Streamlit UI: http://localhost:8501"
    echo "  API (if enabled): http://localhost:8000"
    echo "  Health Check: http://localhost:8501/_stcore/health"
}

# Backup data and models
backup_data() {
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    
    log "Creating backup in $backup_dir..."
    mkdir -p "$backup_dir"
    
    # Backup database
    if docker-compose ps postgres | grep -q "Up"; then
        log "Backing up PostgreSQL database..."
        docker-compose exec postgres pg_dump -U postgres emergency_ai > "$backup_dir/database.sql"
    fi
    
    # Backup models
    log "Backing up models..."
    docker cp emergency-ai-app:/app/models "$backup_dir/"
    
    # Backup logs
    log "Backing up logs..."
    cp -r logs "$backup_dir/" 2>/dev/null || true
    
    # Create backup info
    cat > "$backup_dir/backup_info.json" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "version": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
    "containers": $(docker-compose ps --format json | jq -s .)
}
EOF
    
    success "Backup created: $backup_dir"
}

# Restore from backup
restore_data() {
    local backup_dir="$1"
    
    if [ ! -d "$backup_dir" ]; then
        error "Backup directory not found: $backup_dir"
        exit 1
    fi
    
    log "Restoring from backup: $backup_dir"
    
    # Restore database
    if [ -f "$backup_dir/database.sql" ]; then
        log "Restoring PostgreSQL database..."
        docker-compose exec -T postgres psql -U postgres emergency_ai < "$backup_dir/database.sql"
    fi
    
    # Restore models
    if [ -d "$backup_dir/models" ]; then
        log "Restoring models..."
        docker cp "$backup_dir/models" emergency-ai-app:/app/
    fi
    
    success "Restore completed"
}

# Update deployment
update_deployment() {
    local tag="$1"
    
    log "Updating deployment..."
    
    # Pull latest changes
    if git status &>/dev/null; then
        log "Pulling latest changes from git..."
        git pull
    fi
    
    # Rebuild images
    build_images "$tag" "false"
    
    # Rolling update
    log "Performing rolling update..."
    docker-compose up -d --no-deps emergency-ai
    
    success "Update completed"
}

# Stop services
stop_services() {
    local env="${1:-all}"
    
    log "Stopping services..."
    
    if [ "$env" = "prod" ] || [ "$env" = "all" ]; then
        docker-compose down
    fi
    
    if [ "$env" = "dev" ] || [ "$env" = "all" ]; then
        docker-compose -f docker-compose.dev.yml down
    fi
    
    success "Services stopped"
}

# Restart services  
restart_services() {
    local env="${1:-prod}"
    
    log "Restarting services..."
    
    if [ "$env" = "prod" ]; then
        docker-compose restart
    else
        docker-compose -f docker-compose.dev.yml restart
    fi
    
    success "Services restarted"
}

# Main script logic
main() {
    case "${1:-help}" in
        build)
            check_dependencies
            build_images "$2" "$3"
            ;;
        deploy)
            check_dependencies
            deploy_production
            ;;
        dev)
            check_dependencies
            start_development "$2"
            ;;
        test)
            check_dependencies
            run_tests "$2"
            ;;
        benchmark)
            check_dependencies
            run_benchmarks
            ;;
        clean)
            cleanup
            ;;
        logs)
            show_logs "$2" "$3"
            ;;
        status)
            show_deployment_status
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        update)
            check_dependencies
            update_deployment "$2"
            ;;
        stop)
            stop_services "$2"
            ;;
        restart)
            restart_services "$2"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --env)
            ENV="$2"
            shift 2
            ;;
        --tag)
            DOCKER_TAG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="true"
            shift
            ;;
        --profile)
            PROFILE="$2" 
            shift 2
            ;;
        --verbose)
            VERBOSE="true"
            shift
            ;;
        *)
            COMMAND="$1"
            shift
            ;;
    esac
done

# Run main function
main "$COMMAND" "$ENV" "$DOCKER_TAG" "$NO_CACHE" "$PROFILE" "$VERBOSE"