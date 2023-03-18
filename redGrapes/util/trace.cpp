
#include <chrono>
#include <fstream>
#include <thread>
#include <redGrapes/util/trace.hpp>

std::unique_ptr<perfetto::TracingSession> StartTracing() {
#if REDGRAPES_ENABLE_TRACE
  // The trace config defines which types of data sources are enabled for
  // recording. In this example we just need the "track_event" data source,
  // which corresponds to the TRACE_EVENT trace points.
  perfetto::TraceConfig cfg;
  cfg.add_buffers()->set_size_kb(8192);
  auto* ds_cfg = cfg.add_data_sources()->mutable_config();
  ds_cfg->set_name("track_event");

  auto tracing_session = perfetto::Tracing::NewTrace();
  tracing_session->Setup(cfg);
  tracing_session->StartBlocking();
  return tracing_session;
#else
  return std::unique_ptr<perfetto::TracingSession>();
#endif
}

void StopTracing(std::unique_ptr<perfetto::TracingSession> tracing_session) {
#if REDGRAPES_ENABLE_TRACE
  // Make sure the last event is closed for this example.
  perfetto::TrackEvent::Flush();

  // Stop tracing and read the trace data.
  tracing_session->StopBlocking();
  std::vector<char> trace_data(tracing_session->ReadTraceBlocking());

  // Write the result into a file.
  // Note: To save memory with longer traces, you can tell Perfetto to write
  // directly into a file by passing a file descriptor into Setup() above.
  std::ofstream output;
  output.open("redGrapes.pftrace", std::ios::out | std::ios::binary);
  output.write(&trace_data[0], std::streamsize(trace_data.size()));
  output.close();
#endif
}

