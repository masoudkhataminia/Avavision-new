import Foundation

/// The single JSON format used for stored files, manifests and audit payloads:
/// sorted keys and ISO-8601 dates with fractional seconds.
public enum AvaVisionJSON {
  public static func encoder(pretty: Bool = false) -> JSONEncoder {
    let encoder = JSONEncoder()
    encoder.outputFormatting = pretty ? [.sortedKeys, .prettyPrinted, .withoutEscapingSlashes] : [.sortedKeys]
    encoder.dateEncodingStrategy = .custom { date, encoder in
      var container = encoder.singleValueContainer()
      try container.encode(date.formatted(Date.ISO8601FormatStyle(includingFractionalSeconds: true)))
    }
    return encoder
  }

  public static func decoder() -> JSONDecoder {
    let decoder = JSONDecoder()
    decoder.dateDecodingStrategy = .custom { decoder in
      let container = try decoder.singleValueContainer()
      let string = try container.decode(String.self)
      if let date = try? Date.ISO8601FormatStyle(includingFractionalSeconds: true).parse(string) {
        return date
      }
      if let date = try? Date.ISO8601FormatStyle().parse(string) {
        return date
      }
      throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid ISO-8601 date '\(string)'")
    }
    return decoder
  }
}
