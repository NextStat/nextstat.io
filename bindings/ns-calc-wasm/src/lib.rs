#![doc = "Thin WASM surface for the NextStat A/B Test Sample Size Calculator."]
#![doc = ""]
#![doc = "All statistical logic lives in `ns_inference::calculator`."]
#![doc = "This crate only serialises inputs/outputs across the JS boundary."]

#[cfg(target_arch = "wasm32")]
mod wasm_exports {
    use wasm_bindgen::prelude::*;

    fn js_err(msg: String) -> JsValue {
        JsValue::from_str(&msg)
    }

    #[wasm_bindgen(start)]
    pub fn init() {
        console_error_panic_hook::set_once();
    }

    #[wasm_bindgen]
    pub fn calc_sample_size(config_json: &str) -> Result<JsValue, JsValue> {
        let config: ns_inference::calculator::CalculatorConfig = serde_json::from_str(config_json)
            .map_err(|e| js_err(format!("Invalid config: {e}")))?;
        let result = ns_inference::calculator::calculate_sample_size(&config)
            .map_err(|e| js_err(e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }

    #[wasm_bindgen]
    pub fn calc_power_curve(config_json: &str, num_points: usize) -> Result<JsValue, JsValue> {
        let config: ns_inference::calculator::CalculatorConfig = serde_json::from_str(config_json)
            .map_err(|e| js_err(format!("Invalid config: {e}")))?;
        let pts = num_points.clamp(10, 200);
        let result = ns_inference::calculator::calculate_power_curve(&config, pts)
            .map_err(|e| js_err(e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }

    #[wasm_bindgen]
    pub fn calc_mde_curve(config_json: &str, num_points: usize) -> Result<JsValue, JsValue> {
        let config: ns_inference::calculator::CalculatorConfig = serde_json::from_str(config_json)
            .map_err(|e| js_err(format!("Invalid config: {e}")))?;
        let pts = num_points.clamp(10, 200);
        let result = ns_inference::calculator::calculate_mde_curve(&config, pts)
            .map_err(|e| js_err(e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }

    #[wasm_bindgen]
    pub fn calc_sequential_schedule(config_json: &str) -> Result<JsValue, JsValue> {
        let config: ns_inference::calculator::CalculatorConfig = serde_json::from_str(config_json)
            .map_err(|e| js_err(format!("Invalid config: {e}")))?;
        let result = ns_inference::calculator::calculate_sequential_schedule(&config)
            .map_err(|e| js_err(e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }

    #[wasm_bindgen]
    pub fn calc_sensitivity_breakdown(config_json: &str) -> Result<JsValue, JsValue> {
        let config: ns_inference::calculator::CalculatorConfig = serde_json::from_str(config_json)
            .map_err(|e| js_err(format!("Invalid config: {e}")))?;
        let result = ns_inference::calculator::calculate_sensitivity_breakdown(&config)
            .map_err(|e| js_err(e.to_string()))?;
        serde_wasm_bindgen::to_value(&result).map_err(|e| js_err(e.to_string()))
    }
}
