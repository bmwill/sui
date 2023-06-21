use std::ffi::CStr;

use tracing::info;

/// The package version of the `tsunami` library
pub const PACKAGE_VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PACKAGE_VERSION_CSTR: &std::ffi::CStr = const_str::cstr!(PACKAGE_VERSION);

/// The version of `rustc` used to compile this library
pub const RUSTC_VERSION: &str = env!("RUSTC_VERSION");
pub const RUSTC_VERSION_CSTR: &std::ffi::CStr = const_str::cstr!(RUSTC_VERSION);

/// Declare a `tsunami` plugin type and its constructor.
///
/// # Notes
///
/// This works by automatically generating an `extern "C"` functions with
/// pre-defined signature and symbol names. Therefore you will only be able to
/// declare one plugin per library.
#[macro_export]
macro_rules! declare_plugin {
    ($plugin_type:ty, $constructor:path) => {
        #[no_mangle]
        pub extern "C" fn _package_version() -> *const std::ffi::c_char {
            $crate::PACKAGE_VERSION_CSTR.as_ptr()
        }

        #[no_mangle]
        pub extern "C" fn _rustc_version() -> *const std::ffi::c_char {
            $crate::RUSTC_VERSION_CSTR.as_ptr()
        }

        #[no_mangle]
        #[allow(improper_ctypes_definitions)]
        pub extern "C" fn _plugin_create() -> *mut dyn $crate::Plugin {
            // make sure the constructor is the correct type.
            let constructor: fn() -> $plugin_type = $constructor;

            let object = constructor();
            let boxed: Box<dyn $crate::Plugin> = Box::new(object);
            Box::into_raw(boxed)
        }
    };
}

// #[derive(Debug)]
// pub struct Error {
//     _inner: Box<dyn std::error::Error + Send + Sync>,
// }

pub type Error = Box<dyn std::error::Error + Send + Sync>;

pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Defines a Tsunami plugin, to stream data from the runtime.
/// Tsunami plugins must describe desired behavior for load and unload,
/// as well as how they will handle streamed data.
pub trait Plugin: std::any::Any + Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;

    /// The callback called when a plugin is loaded by the system,
    /// used for doing whatever initialization is required by the plugin.
    fn on_load(&mut self) -> Result<()> {
        Ok(())
    }

    fn trigger(&self, _val: usize) -> Result<()> {
        Ok(())
    }

    /// The callback called right before a plugin is unloaded by the system
    /// Used for doing cleanup before unload.
    fn on_unload(&mut self) {}
}

#[derive(Debug)]
pub struct PluginManager {
    plugins: Vec<PluginType>,
}

impl PluginManager {
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
        }
    }

    /// Unload all plugins and loaded plugin libraries, making sure to fire
    /// their `on_unload()` methods so they can do any necessary cleanup.
    pub fn unload(&mut self) {
        for mut plugin in self.plugins.drain(..) {
            info!("Unloading plugin for {:?}", plugin.plugin().name());
            plugin.plugin_mut().on_unload();
        }
    }

    pub fn plugins(&self) -> impl Iterator<Item = &dyn Plugin> {
        self.plugins.iter().map(PluginType::plugin)
    }

    pub fn load_plugin<P: AsRef<std::path::Path>>(&mut self, filename: P) -> Result<()> {
        use libloading::{Library, Symbol};

        type RustcVersion = unsafe fn() -> *const std::ffi::c_char;
        type PackageVersion = unsafe fn() -> *const std::ffi::c_char;
        type PluginCreate = unsafe fn() -> *mut dyn Plugin;

        let library = unsafe { Library::new(filename.as_ref()) }?;

        let rustc_version: Symbol<RustcVersion> = unsafe { library.get(b"_rustc_version") }?;
        let rustc_version = unsafe { CStr::from_ptr(rustc_version()) };
        if rustc_version != RUSTC_VERSION_CSTR {
            let rustc_version = rustc_version.to_str()?;
            return Err(format!(
                "rustc version does not match. expected {RUSTC_VERSION} found {rustc_version}"
            )
            .into());
        }

        let package_version: Symbol<PackageVersion> = unsafe { library.get(b"_package_version") }?;
        let package_version = unsafe { CStr::from_ptr(package_version()) };
        if package_version != PACKAGE_VERSION_CSTR {
            let package_version = package_version.to_str()?;
            return Err(format!(
                "package version does not match. expected {PACKAGE_VERSION} found {package_version}"
            )
            .into());
        }

        let plugin_create: Symbol<PluginCreate> = unsafe { library.get(b"_plugin_create") }?;
        let boxed_raw = unsafe { plugin_create() };

        let plugin = unsafe { Box::from_raw(boxed_raw) };
        info!("Loaded plugin: {}", plugin.name());

        let mut plugin = PluginType::Dynamic { plugin, library };
        plugin.plugin_mut().on_load()?;
        self.plugins.push(plugin);

        Ok(())
    }

    pub fn load_static_plugin<P: Plugin>(&mut self, plugin: P) -> Result<()> {
        let mut plugin = PluginType::Static {
            plugin: Box::new(plugin),
        };
        plugin.plugin_mut().on_load()?;
        self.plugins.push(plugin);

        Ok(())
    }
}

impl Drop for PluginManager {
    fn drop(&mut self) {
        self.unload();
    }
}

#[derive(Debug)]
enum PluginType {
    /// Dynamiclly Loaded/Linked Plugin
    Dynamic {
        plugin: Box<dyn Plugin>,
        #[allow(unused)]
        library: libloading::Library,
    },

    /// Staticlly Linked Plugin
    Static { plugin: Box<dyn Plugin> },
}

impl PluginType {
    fn plugin(&self) -> &dyn Plugin {
        match self {
            PluginType::Dynamic { plugin, .. } | PluginType::Static { plugin } => &**plugin,
        }
    }

    fn plugin_mut(&mut self) -> &mut dyn Plugin {
        match self {
            PluginType::Dynamic { plugin, .. } | PluginType::Static { plugin } => &mut **plugin,
        }
    }
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[derive(Default, Debug)]
    struct TestPlugin;

    impl Plugin for TestPlugin {
        fn name(&self) -> &'static str {
            "test"
        }
    }

    declare_plugin!(TestPlugin, TestPlugin::default);

    #[test]
    fn test_plugin() {
        let pkg_version = unsafe { std::ffi::CStr::from_ptr(_package_version()) };
        assert_eq!(pkg_version, PACKAGE_VERSION_CSTR);
        let rust_version = unsafe { std::ffi::CStr::from_ptr(_rustc_version()) };
        assert_eq!(rust_version, RUSTC_VERSION_CSTR);

        let plugin: Box<dyn Plugin> = unsafe { Box::from_raw(_plugin_create()) };
        assert_eq!(plugin.name(), "test");
    }
}
