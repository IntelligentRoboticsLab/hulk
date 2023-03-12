use proc_macro::TokenStream;
use quote::quote;
/*
For Rust learners:

This is a Rust function that uses the Rust macro system to generate code at compile time. 
Here's how it works:

1. The function takes a TokenStream as input. 
This is a representation of Rust code that has been parsed into individual tokens.

2. The first line of the function converts the TokenStream input into a proc_macro2::TokenStream. 
This is done so that we can use the quote! macro from the quote crate to generate Rust code.

3. The quote! macro is then used to generate a Rust match expression. 
This match expression takes the input field (which is expected to be an Option) and 
matches on its value. If field is Some, the value inside is returned. 
If field is None, the function returns an Ok result with a value of MainOutputs::none().

4. Finally, the generated Rust code is converted back into a TokenStream using the into() method, 
and returned from the function.

In summary, this function takes an Option value as input, 
and returns either the value inside the Some variant, 
or an Ok result with a default value if the input is None. 
The code generation is done using the Rust macro system, 
which allows for automatic code generation at compile time.

Credits: ChatGPT
 */
pub fn process_require_some(input: TokenStream) -> TokenStream {
    let field = proc_macro2::TokenStream::from(input);
    let expanded = quote! {
        match #field {
            Some(data) => data,
            None => return Ok(MainOutputs::none()),
        }
    };
    expanded.into()
}
