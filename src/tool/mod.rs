//! Tool trait: functions that agents can invoke.

use comp_cat_rs::effect::io::Io;
use serde_json::Value;

use crate::error::Error;

/// A tool definition: metadata sent to the LLM so it knows
/// what tools are available and how to call them.
#[derive(Debug, Clone)]
pub struct ToolDefinition {
    name: String,
    description: String,
    parameters_schema: Value,
}

impl ToolDefinition {
    #[must_use]
    pub fn new(name: String, description: String, parameters_schema: Value) -> Self {
        Self { name, description, parameters_schema }
    }

    #[must_use]
    pub fn name(&self) -> &str { &self.name }

    #[must_use]
    pub fn description(&self) -> &str { &self.description }

    #[must_use]
    pub fn parameters_schema(&self) -> &Value { &self.parameters_schema }

    /// Format as the JSON structure expected by LLM tool-calling APIs.
    #[must_use]
    pub fn to_api_json(&self) -> Value {
        serde_json::json!({
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema
            }
        })
    }
}

/// A tool that an agent can call during reasoning.
///
/// Tools receive JSON arguments and return a JSON result,
/// wrapped in `Io` for effect tracking.
pub trait Tool {
    /// The tool's definition (name, description, schema).
    fn definition(&self) -> ToolDefinition;

    /// Execute the tool with the given JSON arguments.
    fn call(&self, args: Value) -> Io<Error, Value>;
}

/// A collection of tools available to an agent.
///
/// Generic over the tool type `T` to avoid `dyn Trait`.
/// All tools in a toolbox must be the same concrete type.
/// For heterogeneous tools, use an enum that implements `Tool`.
///
/// ```rust,ignore
/// enum MyTools {
///     Calculator(Calculator),
///     WebSearch(WebSearch),
/// }
///
/// impl Tool for MyTools {
///     fn definition(&self) -> ToolDefinition {
///         match self {
///             Self::Calculator(t) => t.definition(),
///             Self::WebSearch(t) => t.definition(),
///         }
///     }
///     fn call(&self, args: Value) -> Io<Error, Value> {
///         match self {
///             Self::Calculator(t) => t.call(args),
///             Self::WebSearch(t) => t.call(args),
///         }
///     }
/// }
/// ```
pub struct Toolbox<T: Tool> {
    tools: Vec<T>,
}

impl<T: Tool> Toolbox<T> {
    /// Create an empty toolbox.
    #[must_use]
    pub fn new() -> Self { Self { tools: Vec::new() } }

    /// Add a tool to the toolbox.
    #[must_use]
    pub fn with_tool(self, tool: T) -> Self {
        Self {
            tools: self.tools.into_iter().chain(std::iter::once(tool)).collect(),
        }
    }

    /// Look up a tool by name and invoke it.
    #[must_use]
    pub fn invoke(&self, name: &str, args: Value) -> Io<Error, Value> {
        let name_owned = name.to_owned();
        self.tools.iter()
            .find(|t| t.definition().name() == name_owned)
            .map_or_else(
                move || Io::suspend(move || {
                    Err(Error::Config { field: format!("unknown tool: {name_owned}") })
                }),
                |t| t.call(args),
            )
    }

    /// Get tool definitions for sending to the LLM.
    #[must_use]
    pub fn definitions(&self) -> Vec<Value> {
        self.tools.iter().map(|t| t.definition().to_api_json()).collect()
    }
}

impl<T: Tool> Default for Toolbox<T> {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeTool {
        name: String,
    }

    impl Tool for FakeTool {
        fn definition(&self) -> ToolDefinition {
            ToolDefinition::new(
                self.name.clone(),
                "a fake tool".into(),
                serde_json::json!({"type": "object"}),
            )
        }

        fn call(&self, _args: Value) -> Io<Error, Value> {
            Io::pure(serde_json::json!({"result": "ok"}))
        }
    }

    #[test]
    fn invoke_known_tool_succeeds() -> Result<(), Error> {
        let toolbox = Toolbox::new()
            .with_tool(FakeTool { name: "greet".into() });
        let result = toolbox.invoke("greet", serde_json::json!({})).run()?;
        assert_eq!(result, serde_json::json!({"result": "ok"}));
        Ok(())
    }

    #[test]
    fn invoke_unknown_tool_returns_error() {
        let toolbox: Toolbox<FakeTool> = Toolbox::new();
        let result = toolbox.invoke("nonexistent", serde_json::json!({})).run();
        assert!(result.is_err());
    }

    #[test]
    fn definitions_lists_all_tools() {
        let toolbox = Toolbox::new()
            .with_tool(FakeTool { name: "a".into() })
            .with_tool(FakeTool { name: "b".into() });
        let defs = toolbox.definitions();
        assert_eq!(defs.len(), 2);
    }
}
